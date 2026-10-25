"""Context management: keep the prompt small as a trajectory grows.

Strategies (AgentConfig.compression):
- "none":      send the full history every step (baseline).
- "truncate":  keep the last `keep_last` tool results verbatim; older tool results are cut to
               `old_chars` characters with a marker, so the model still sees what it did.
- "summarize": older tool results are replaced by one short summary written by a (cheap) model.
"""
from __future__ import annotations

import copy


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count (4 characters per token). Billing always uses the provider's reported usage."""
    n = 0
    for m in messages:
        n += 4 + len(str(m.get("content") or "")) // 4
        for c in m.get("tool_calls") or []:
            n += len(c["function"]["arguments"] or "") // 4 + 4
    return n


def _tool_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "tool"]


def truncate_old_tool_results(messages: list[dict], keep_last: int = 2, old_chars: int = 200) -> list[dict]:
    out = copy.deepcopy(messages)
    idx = _tool_indices(out)
    for i in idx[:-keep_last] if keep_last else idx:
        c = out[i]["content"]
        if len(c) > old_chars:
            out[i]["content"] = c[:old_chars] + f" ...[older result compressed, {len(c) - old_chars} chars dropped]"
    return out


def summarize_old_tool_results(messages: list[dict], summarizer, keep_last: int = 2,
                               max_tokens: int = 200) -> tuple[list[dict], object | None]:
    """Replace older tool results with a single summary. Returns (messages, summarizer Completion or None)
    so the caller can bill the summarizer call."""
    idx = _tool_indices(messages)
    old = idx[:-keep_last] if keep_last else idx
    if not old:
        return messages, None
    notes = "\n\n".join(f"[{i}] {messages[i]['content'][:2000]}" for i in old)
    comp = summarizer.complete([
        {"role": "system", "content": "Summarize these tool results into the facts needed to answer the task. "
                                      "Be terse; keep names, numbers and quotes exactly."},
        {"role": "user", "content": notes}], max_tokens=max_tokens)
    out = copy.deepcopy(messages)
    for i in old:
        out[i]["content"] = "[summarized below]"
    out[old[-1]]["content"] = "Summary of earlier tool results: " + comp.text.strip()
    return out, comp
