#!/bin/zsh
# Feature 1 experiment: every context strategy on the same seeded HotpotQA-Session tasks.
# Needs ANTHROPIC_API_KEY and DEEPSEEK_API_KEY in the environment.
cd "$(dirname "$0")/.."
B=hotpotqa_session
for c in haiku_nocache_full haiku_full haiku_truncate haiku_checkpoint haiku_checkpoint8k; do
  python3 scripts/run_bench.py --config configs/cache/$c.toml --bench $B --n 30 --concurrency 6 --name exp_$c
done
for c in deepseek_flash_full deepseek_flash_truncate deepseek_flash_checkpoint deepseek_flash_checkpoint8k; do
  python3 scripts/run_bench.py --config configs/cache/$c.toml --bench $B --n 30 --concurrency 6 --name exp_$c
done
for c in sonnet_full sonnet_truncate sonnet_checkpoint sonnet_checkpoint8k; do
  python3 scripts/run_bench.py --config configs/cache/$c.toml --bench $B --n 20 --concurrency 5 --name exp_$c
done
echo EXPERIMENT_DONE
