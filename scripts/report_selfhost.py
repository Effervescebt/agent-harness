"""Markdown tables for the self-hosted serving experiment (results/exp_qwen_*), compared with the API runs.

  python scripts/report_selfhost.py > results/selfhost_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"


def load(name):
    d = ROOT / name
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    m = d / "vllm_metrics.json"
    s["vllm"] = json.loads(m.read_text()) if m.exists() else {}
    return s


print("### Self-hosted Qwen3-8B on vLLM (one NVIDIA L40, $0.69/h), HotpotQA-Session\n")
print("| Run | Sessions | Concurrency | Question EM (95% CI) | GPU $ / session | Server KV-cache hit | "
      "Sessions / GPU-hour | p50 session latency (s) | Steps |")
print("|---|---|---|---|---|---|---|---|---|")
for name, label in [("exp_qwen_full", "Full history, prefix cache on"),
                    ("exp_qwen_nocache_full", "Full history, prefix cache off (salted)"),
                    ("exp_qwen_truncate", "Rolling truncation, cache on"),
                    ("exp_qwen_checkpoint8k", "Checkpoint compaction (8K), cache on"),
                    ("exp_qwen_full_c1", "Full history, cache on"),
                    ("exp_qwen_full_c16", "Full history, cache on")]:
    s = load(name)
    if not s:
        continue
    lo, hi = s["score_ci95"]
    hit = s["vllm"].get("hit_rate")
    print(f"| {label} | {s['n']} | {s['concurrency']} | {s['mean_score']:.3f} [{lo:.3f}, {hi:.3f}] | "
          f"{s['gpu_cost_per_task_usd']:.5f} | {'' if hit is None else f'{100 * hit:.1f}%'} | "
          f"{s['sessions_per_gpu_hour']:.0f} | {s['latency_p50_s']:.1f} | {s['mean_steps']:.1f} |")

print("\n### Same benchmark through APIs (full history, caching on)\n")
print("| Model | Question EM | $ / session |\n|---|---|---|")
for name, label in [("exp_haiku_full", "Claude Haiku 5.5"), ("exp_deepseek_flash_full", "DeepSeek (deepseek-flash)"),
                    ("exp_sonnet_full", "Claude Sonnet 5.5")]:
    s = load(name)
    if s:
        print(f"| {label} | {s['mean_score']:.3f} | {s['cost_per_task_usd']:.5f} |")
