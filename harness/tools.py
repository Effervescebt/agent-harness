"""Tool registry with JSON-schema validation of model-produced arguments.

A tool is a Python function plus a JSON schema for its arguments. The agent never calls a tool with
arguments that fail validation; it returns the validation error to the model instead, so the model can
repair the call (malformed-output recovery).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "object": dict, "array": list, "null": type(None)}


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate against the subset of JSON Schema that tool definitions use:
    type, properties, required, additionalProperties, enum, items, minimum, maximum, minLength, maxLength."""
    errors: list[str] = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(isinstance(value, _TYPES[x]) and not (x in ("integer", "number") and isinstance(value, bool))
                 for x in types if x in _TYPES)
        if not ok:
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}: missing required field '{k}'")
        for k, v in value.items():
            if k in props:
                errors += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field '{k}'")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += validate(v, schema["items"], f"{path}[{i}]")
    return errors


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    side_effect_free: bool = True  # only side-effect-free tools may run in parallel

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolResult:
    ok: bool
    content: str
    latency_s: float
    error_kind: str = ""  # "", "parse", "schema", "unknown_tool", "exception", "timeout"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, max_output_chars: int = 4000):
        self.tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def parse(self, name: str, raw_args: str) -> tuple[dict | None, ToolResult | None]:
        """Return (args, None) when the call is usable, else (None, error result for the model)."""
        if name not in self.tools:
            return None, ToolResult(False, f"Error: unknown tool '{name}'. Available: {sorted(self.tools)}",
                                    0.0, "unknown_tool")
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return None, ToolResult(False, f"Error: arguments are not valid JSON ({e.msg}). "
                                           f"Resend the call with a JSON object.", 0.0, "parse")
        errs = validate(args, self.tools[name].parameters)
        if errs:
            return None, ToolResult(False, "Error: invalid arguments: " + "; ".join(errs[:5]), 0.0, "schema")
        return args, None

    def run(self, name: str, args: dict) -> ToolResult:
        t0 = time.perf_counter()
        try:
            out = self.tools[name].fn(**args)
            text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            if len(text) > self.max_output_chars:
                text = text[: self.max_output_chars] + f"\n...[truncated {len(text) - self.max_output_chars} chars]"
            return ToolResult(True, text, time.perf_counter() - t0)
        except Exception as e:  # tool bugs are reported to the model, never crash the run
            return ToolResult(False, f"Error: tool raised {type(e).__name__}: {e}", time.perf_counter() - t0,
                              "exception")


FINAL_ANSWER = Tool(
    name="final_answer",
    description="Submit the final answer and stop. Call this exactly once, when you are confident.",
    parameters={"type": "object", "properties": {"answer": {"type": "string", "minLength": 1}},
                "required": ["answer"], "additionalProperties": False},
    fn=lambda answer: answer,
)
