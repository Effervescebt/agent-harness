"""Markdown tables for the caching experiment (results/exp_*), one table per model.

  python scripts/report_cache.py > results/cache_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
LABELS = {"nocache_full": "Full history, caching off", "full": "Full history", "truncate": "Rolling truncation",
          "checkpoint": "Checkpoint compaction (2K chars)", "checkpoint8k": "Checkpoint compaction (8K chars)"}
MODELS = {"haiku": "Claude Haiku 5.5", "deepseek_flash": "DeepSeek (deepseek-flash)", "sonnet": "Claude Sonnet 5.5"}

for key, title in MODELS.items():
    runs = [(strat, ROOT / f"exp_{key}_{strat}") for strat in LABELS]
    runs = [(st, d) for st, d in runs if (d / "summary.json").exists()]
    if not runs:
        continue
    rows = [(st, json.loads((d / "summary.json").read_text())) for st, d in runs]
    base = dict(rows).get("full") or rows[0][1]
    print(f"\n### {title} (n = {rows[0][1]['n']} sessions x 6 questions)\n")
    print("| Strategy | Question EM (95% CI) | $ / session | vs full | Cache hit | Prefix breaks / session "
          "| Prompt tokens / session | p50 latency (s) | Steps |")
    print("|---|---|---|---|---|---|---|---|---|")
    for st, s in rows:
        lo, hi = s["score_ci95"]
        d = (s["cost_per_task_usd"] - base["cost_per_task_usd"]) / base["cost_per_task_usd"] * 100
        print(f"| {LABELS[st]} | {s['mean_score']:.3f} [{lo:.3f}, {hi:.3f}] | {s['cost_per_task_usd']:.5f} | "
              f"{d:+.1f}% | {100 * s['cache_hit_rate']:.1f}% | {s['prefix_breaks_per_task']:.1f} | "
              f"{s['mean_input_tokens']:,.0f} | {s['latency_p50_s']:.1f} | {s['mean_steps']:.1f} |")
