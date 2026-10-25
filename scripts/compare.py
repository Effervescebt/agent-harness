"""Compare runs side by side; the first run is the baseline for the delta columns.

  python scripts/compare.py results/<baseline run> results/<run 2> [...]
"""
import json
import sys
from pathlib import Path

runs = [Path(p) for p in sys.argv[1:]]
if len(runs) < 1:
    sys.exit(__doc__)
rows = [(r.name, json.loads((r / "summary.json").read_text())) for r in runs]
base = rows[0][1]


def pct(v, b):
    return "" if not b else f"{(v - b) / b * 100:+.1f}%"


print(f"{'run':34s} {'n':>3s} {'score':>6s} {'95% CI':>15s} {'$/task':>9s} {'d$':>7s} {'hit%':>5s} "
      f"{'brk':>4s} {'p50 s':>6s} {'dp50':>7s} {'steps':>5s} {'in tok':>7s} {'d tok':>7s}")
for name, s in rows:
    lo, hi = s.get("score_ci95", s["success_ci95"])
    print(f"{name[:34]:34s} {s['n']:3d} {s['mean_score']:6.3f} {f'[{lo:.3f}, {hi:.3f}]':>15s} "
          f"{s['cost_per_task_usd']:9.5f} {pct(s['cost_per_task_usd'], base['cost_per_task_usd']):>7s} "
          f"{100 * s.get('cache_hit_rate', 0):5.1f} {s.get('prefix_breaks_per_task', 0):4.1f} "
          f"{s['latency_p50_s']:6.1f} {pct(s['latency_p50_s'], base['latency_p50_s']):>7s} "
          f"{s['mean_steps']:5.1f} {s['mean_input_tokens']:7.0f} {pct(s['mean_input_tokens'], base['mean_input_tokens']):>7s}")
