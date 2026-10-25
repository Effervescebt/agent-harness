"""Cost, latency and reliability metrics for a set of trajectories.

Prices come from configs/models.toml (USD per 1M tokens). Cost uses the token counts the provider
reports, so it matches the bill; cached input tokens are charged at the cached rate when one is given.
"""
from __future__ import annotations

import math
import random
import statistics


def call_cost(model: str, usage: dict, prices: dict) -> float:
    """input_tokens is the total prompt; cached reads and cache writes are priced separately
    (writes default to the plain input price, i.e. no premium, for providers without one)."""
    p = prices.get(model)
    if not p:
        return 0.0
    cached = usage.get("cached_input_tokens", 0) or 0
    write = usage.get("cache_write_tokens", 0) or 0
    fresh = max(0, usage.get("input_tokens", 0) - cached - write)
    return (fresh * p["input"] + cached * p.get("cached_input", p["input"])
            + write * p.get("cache_write", p["input"]) + usage.get("output_tokens", 0) * p["output"]) / 1e6


def trajectory_cost(traj: dict, prices: dict) -> float:
    total = sum(call_cost(s["model"], s["usage"], prices) for s in traj["steps"])
    return total + sum(call_cost(c["model"], c["usage"], prices) for c in traj.get("extra_calls", []))


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_ci(xs: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean, so small differences between configs are not
    over-claimed."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters) - 1]


def summarize(records: list[dict], prices: dict) -> dict:
    """records: one dict per task with keys 'trajectory' (or 'attempts' for cascades), 'correct', 'score'."""
    n = len(records)
    if not n:
        return {"n": 0}
    costs, lat, steps, in_tok, out_tok, cached_tok, write_tok = [], [], [], [], [], [], []
    tool_calls = tool_errors = repairs = escalated = 0
    status: dict[str, int] = {}
    for r in records:
        attempts = r.get("attempts") or [r["trajectory"]]
        escalated += len(attempts) > 1
        costs.append(sum(trajectory_cost(t, prices) for t in attempts))
        lat.append(sum(t["wall_s"] for t in attempts))
        steps.append(sum(len(t["steps"]) for t in attempts))
        in_tok.append(sum(s["usage"]["input_tokens"] for t in attempts for s in t["steps"]))
        out_tok.append(sum(s["usage"]["output_tokens"] for t in attempts for s in t["steps"]))
        cached_tok.append(sum(s["usage"].get("cached_input_tokens", 0) for t in attempts for s in t["steps"]))
        write_tok.append(sum(s["usage"].get("cache_write_tokens", 0) for t in attempts for s in t["steps"]))
        for t in attempts:
            for s in t["steps"]:
                repairs += s["repaired"]
                for c in s["tool_calls"]:
                    tool_calls += 1
                    tool_errors += not c["ok"]
        final = attempts[-1]["status"]
        status[final] = status.get(final, 0) + 1
    correct = [1.0 if r["correct"] else 0.0 for r in records]
    lo, hi = bootstrap_ci(correct)
    return {
        "n": n,
        "success_rate": statistics.fmean(correct),
        "success_ci95": [lo, hi],
        "mean_score": statistics.fmean(r["score"] for r in records),
        "score_ci95": list(bootstrap_ci([r["score"] for r in records])),
        "cost_per_task_usd": statistics.fmean(costs),
        "total_cost_usd": sum(costs),
        "cost_per_success_usd": sum(costs) / max(1, sum(correct)),
        "latency_p50_s": percentile(lat, 0.5),
        "latency_p95_s": percentile(lat, 0.95),
        "mean_steps": statistics.fmean(steps),
        "mean_input_tokens": statistics.fmean(in_tok),
        "mean_output_tokens": statistics.fmean(out_tok),
        "cache_hit_rate": sum(cached_tok) / max(1, sum(in_tok)),   # share of prompt tokens read from cache
        "cache_write_rate": sum(write_tok) / max(1, sum(in_tok)),
        "tool_error_rate": tool_errors / max(1, tool_calls),
        "repair_steps": repairs,
        "escalation_rate": escalated / n,
        "prefix_breaks_per_task": statistics.fmean(sum(t.get("prefix_breaks", 0) for t in (r.get("attempts") or [r["trajectory"]])) for r in records),
        "compactions_per_task": statistics.fmean(sum(t.get("compactions", 0) for t in (r.get("attempts") or [r["trajectory"]])) for r in records),
        "status": status,
    }
