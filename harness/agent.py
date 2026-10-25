"""The agent loop.

One task = one trajectory: the model is called step by step; each step it either calls tools or answers.
The loop owns everything around the model:

- schema-validated tool calling (native function calling, or a JSON-in-text fallback for models without it)
- malformed-output recovery: invalid JSON, unknown tools or schema errors go back to the model as tool
  errors, up to `max_repairs` times
- termination: a `final_answer` tool call, a step budget, and an optional forced answer on the last step
- parallel execution of independent, side-effect-free tool calls issued in the same step
- context compression of old tool results (see context.py), including cache-preserving checkpoint
  compaction: history stays append-only between rare, batched compaction events, so provider prompt
  caches keep hitting
- prefix-break accounting: a step whose prompt is not the previous prompt plus appended messages
  invalidates every provider cache after the changed message; counted per trajectory
- per-step accounting of tokens, latency and tool errors for the metrics module
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .context import estimate_tokens, summarize_old_tool_results, truncate_old_tool_results
from .llm import Completion, LLMClient, LLMError, ToolCall, Usage
from .tools import FINAL_ANSWER, ToolRegistry, ToolResult


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tokens_per_step: int = 1024
    temperature: float = 0.0
    parallel_tools: bool = False        # run independent tool calls of one step concurrently
    max_parallel: int = 8
    compression: str = "none"           # "none" | "truncate" | "summarize" | "checkpoint"
    keep_last: int = 2
    old_chars: int = 200
    checkpoint_chars: int = 2000        # checkpoint: compact once this many chars of old tool output pile up
    max_repairs: int = 3                # malformed outputs tolerated before giving up
    force_final_answer: bool = True     # on the last step, only final_answer is offered
    text_tool_mode: bool = False        # JSON-in-text tool calls for models without native tool calling
    system_prompt: str = ("You are a careful agent. Use the tools to gather evidence, then call final_answer "
                          "with a short answer. Call several independent tools in one step when possible.")


@dataclass
class StepRecord:
    step: int
    model: str
    usage: Usage
    latency_s: float
    prompt_tokens_est: int
    tool_calls: list[dict] = field(default_factory=list)  # name, args, ok, error_kind, latency_s
    repaired: bool = False


@dataclass
class Trajectory:
    task_id: str
    answer: str | None = None
    status: str = "running"  # answered | max_steps | repairs_exhausted | llm_error
    steps: list[StepRecord] = field(default_factory=list)
    extra_calls: list[dict] = field(default_factory=list)  # e.g. summarizer calls, billed separately
    wall_s: float = 0.0
    error: str = ""
    prefix_breaks: int = 0   # steps whose prompt rewrote (not just extended) the previous prompt
    compactions: int = 0     # checkpoint compaction events

    def to_dict(self) -> dict:
        return asdict(self)


TEXT_MODE_INSTRUCTIONS = """You can call tools. To call tools, reply with ONLY a JSON object, no other text:
{"tool_calls": [{"name": "<tool name>", "arguments": {...}}, ...]}
Available tools (JSON schemas):
%s
To finish, call the tool "final_answer"."""


def parse_text_tool_calls(text: str) -> list[ToolCall] | None:
    """Extract {"tool_calls": [...]} from free text (handles ```json fences and surrounding prose).
    Returns None when no parseable object is present."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    calls = obj.get("tool_calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list):
        return None
    out = []
    for i, c in enumerate(calls):
        if isinstance(c, dict) and "name" in c:
            out.append(ToolCall(f"text_{uuid.uuid4().hex[:8]}_{i}", str(c["name"]),
                                json.dumps(c.get("arguments", {}))))
    return out or None


class Agent:
    def __init__(self, client: LLMClient, tools: ToolRegistry, config: AgentConfig | None = None,
                 summarizer: LLMClient | None = None, model_for_step=None):
        self.client, self.config = client, config or AgentConfig()
        self.tools = tools
        if "final_answer" not in self.tools.tools:
            self.tools.add(FINAL_ANSWER)
        self.summarizer = summarizer or client
        self.model_for_step = model_for_step  # optional router: (step, traj) -> LLMClient

    # ---- message building -------------------------------------------------------------------
    def _system(self) -> str:
        if self.config.text_tool_mode:
            return self.config.system_prompt + "\n\n" + TEXT_MODE_INSTRUCTIONS % json.dumps(
                self.tools.schemas(), indent=1)
        return self.config.system_prompt

    def _checkpoint(self, messages: list[dict], traj: Trajectory) -> None:
        """Cache-preserving compaction: rewrite history in place, but only in batches. Old tool results
        (all but the last keep_last) accumulate verbatim until they exceed checkpoint_chars, then all of
        them are compressed at once; between events the history is append-only."""
        c = self.config
        idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        old = [i for i in (idx[:-c.keep_last] if c.keep_last else idx)
               if len(messages[i]["content"]) > c.old_chars and not messages[i].get("_compacted")]
        if sum(len(messages[i]["content"]) for i in old) < c.checkpoint_chars:
            return
        for i in old:
            text = messages[i]["content"]
            messages[i] = {**messages[i], "_compacted": True,
                           "content": text[:c.old_chars] + f" ...[compacted, {len(text) - c.old_chars} chars dropped]"}
        traj.compactions += 1

    @staticmethod
    def _clean(view: list[dict]) -> list[dict]:
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in view]

    def _view(self, messages: list[dict], traj: Trajectory) -> list[dict]:
        c = self.config
        view = messages
        if c.compression == "checkpoint":
            self._checkpoint(messages, traj)
            view = self._clean(messages)
        elif c.compression == "truncate":
            view = truncate_old_tool_results(messages, c.keep_last, c.old_chars)
        elif c.compression == "summarize":
            view, comp = summarize_old_tool_results(messages, self.summarizer, c.keep_last)
            if comp is not None:
                traj.extra_calls.append({"kind": "summarize", "model": comp.model, "usage": asdict(comp.usage),
                                         "latency_s": comp.latency_s})
        return self._as_text(view) if c.text_tool_mode else view

    @staticmethod
    def _as_text(messages: list[dict]) -> list[dict]:
        """Text mode: models without native tools reject 'tool' roles, so replay calls and results as text."""
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                         for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": json.dumps({"tool_calls": calls})})
            elif m["role"] == "tool":
                text = f"Result of {m.get('name', 'tool')}:\n{m['content']}"
                if out and out[-1]["role"] == "user":
                    out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + text}
                else:
                    out.append({"role": "user", "content": text})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    # ---- tool execution ---------------------------------------------------------------------
    def _execute(self, calls: list[ToolCall]) -> list[tuple[ToolCall, dict | None, ToolResult]]:
        parsed = [(c, *self.tools.parse(c.name, c.arguments)) for c in calls]
        todo = [(i, c, a) for i, (c, a, err) in enumerate(parsed) if err is None and c.name != "final_answer"]
        results: dict[int, ToolResult] = {i: err for i, (c, a, err) in enumerate(parsed) if err is not None}
        parallel = (self.config.parallel_tools and len(todo) > 1
                    and all(self.tools.tools[c.name].side_effect_free for _, c, _ in todo))
        if parallel:
            with ThreadPoolExecutor(max_workers=min(self.config.max_parallel, len(todo))) as ex:
                futs = {i: ex.submit(self.tools.run, c.name, a) for i, c, a in todo}
                results.update({i: f.result() for i, f in futs.items()})
        else:
            for i, c, a in todo:
                results[i] = self.tools.run(c.name, a)
        for i, (c, a, err) in enumerate(parsed):
            if c.name == "final_answer" and err is None:
                results[i] = ToolResult(True, a["answer"], 0.0)
        return [(c, a, results[i]) for i, (c, a, _) in enumerate(parsed)]

    # ---- main loop --------------------------------------------------------------------------
    def run(self, task_id: str, user_prompt: str) -> Trajectory:
        cfg, traj = self.config, Trajectory(task_id)
        messages: list[dict] = [{"role": "system", "content": self._system()},
                                {"role": "user", "content": user_prompt}]
        repairs, t_start, prev_view = 0, time.perf_counter(), None
        for step in range(cfg.max_steps):
            last = step == cfg.max_steps - 1
            schemas = self.tools.schemas()
            if last and cfg.force_final_answer:
                schemas = [FINAL_ANSWER.schema()]
                messages.append({"role": "user", "content": "Step budget reached. Call final_answer now "
                                                            "with your best answer."})
            view = self._view(messages, traj)
            if prev_view is not None and view[:len(prev_view)] != prev_view:
                traj.prefix_breaks += 1
            prev_view = json.loads(json.dumps(view))
            client = self.model_for_step(step, traj) if self.model_for_step else self.client
            try:
                comp: Completion = client.complete(view, None if cfg.text_tool_mode else schemas,
                                                   cfg.max_tokens_per_step, cfg.temperature)
            except LLMError as e:
                traj.status, traj.error = "llm_error", str(e)
                break
            rec = StepRecord(step, comp.model, comp.usage, comp.latency_s, estimate_tokens(view))
            traj.steps.append(rec)
            calls = comp.tool_calls
            if cfg.text_tool_mode:
                calls = parse_text_tool_calls(comp.text) or []

            if not calls:  # no tool call: malformed for a tool-using agent
                repairs += 1
                rec.repaired = True
                messages.append({"role": "assistant", "content": comp.text or ""})
                if repairs > cfg.max_repairs:
                    traj.status = "repairs_exhausted"
                    break
                messages.append({"role": "user", "content": "You must call a tool. If you know the answer, "
                                                            "call final_answer."})
                continue

            messages.append({"role": "assistant", "content": comp.text or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name, "arguments": c.arguments}}
                                            for c in calls]})
            answered = None
            for call, args, res in self._execute(calls):
                rec.tool_calls.append({"name": call.name, "args": args, "ok": res.ok,
                                       "error_kind": res.error_kind, "latency_s": res.latency_s})
                if not res.ok:
                    repairs += 1
                    rec.repaired = True
                if call.name == "final_answer" and res.ok:
                    answered = res.content
                content = res.content if call.name != "final_answer" else "Answer recorded."
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
            if answered is not None:
                traj.answer, traj.status = answered, "answered"
                break
            if repairs > cfg.max_repairs:
                traj.status = "repairs_exhausted"
                break
        else:
            traj.status = "max_steps"
        if traj.status == "running":
            traj.status = "max_steps"
        traj.wall_s = time.perf_counter() - t_start
        return traj
