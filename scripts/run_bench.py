"""Run one benchmark under one config.

  python scripts/run_bench.py --config configs/baseline.toml --bench hotpotqa --n 100
  python scripts/run_bench.py --config configs/baseline.toml --bench gsm8k --n 20 --dry-run   # mock model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.runner import run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--bench", required=True, choices=["hotpotqa", "hotpotqa_session", "gsm8k"])
p.add_argument("--n", type=int, default=50, help="number of tasks (0 = all)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--concurrency", type=int, default=4, help="tasks run in parallel")
p.add_argument("--name", help="results folder name")
p.add_argument("--dry-run", action="store_true", help="use the mock model for every role (no API calls)")
a = p.parse_args()
over = {}
if a.dry_run:
    over = {"routing.primary": "mock", "routing.strong": "mock", "routing.summarizer": "mock"}
run(a.config, a.bench, a.n, a.seed, a.concurrency, a.name, over)
