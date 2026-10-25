"""A deterministic stand-in "model" for dry runs (provider = "mock").

It follows a fixed recipe (search -> read -> answer, or calculator -> answer) so the whole pipeline -
tools, validation, compression, metrics, result files - can be exercised without an API key. Its answers
are not meant to be good; dry-run accuracy says nothing about real models.
"""
from __future__ import annotations

import json
import re

from .llm import Completion, ToolCall, Usage


def _call(name: str, **args) -> Completion:
    return Completion("", [ToolCall(f"mock_{name}_{abs(hash(json.dumps(args))) % 10**8}", name, json.dumps(args))],
                      Usage(), 0.0, "mock")


def scripted_policy(messages: list[dict], tools: list[dict] | None) -> Completion:
    names = {t["name"] for t in tools or []}
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    question = next((m["content"] for m in messages if m["role"] == "user"), "")
    if names == {"final_answer"}:
        return _call("final_answer", answer="unknown")
    if "search" in names:
        if not tool_msgs:
            q = question.split("Question:")[-1].strip()[:200]
            return _call("search", query=q)
        if len(tool_msgs) == 1:
            try:
                title = json.loads(tool_msgs[0]["content"])[0]["title"]
            except (ValueError, KeyError, IndexError):
                title = "unknown"
            return _call("read", title=title)
        words = tool_msgs[-1]["content"].split()[:3]
        return _call("final_answer", answer=" ".join(words) or "unknown")
    if "calculator" in names:
        nums = re.findall(r"\d+\.?\d*", question.split("Problem:")[-1])
        if not tool_msgs and len(nums) >= 2:
            return _call("calculator", expression=f"{nums[0]} + {nums[1]}")
        return _call("final_answer", answer=tool_msgs[-1]["content"] if tool_msgs else (nums[0] if nums else "0"))
    return _call("final_answer", answer="unknown")
