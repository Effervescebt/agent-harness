#!/bin/zsh
# Self-hosted serving experiment on one vLLM server (Qwen3-8B, L40). Needs VLLM_BASE_URL, VLLM_API_KEY.
# Each run records vLLM's prefix-cache counters before and after (results/<run>/vllm_metrics.json).
cd "$(dirname "$0")/.."
B=hotpotqa_session
run() {  # run <config> <name> <n> <concurrency>
  before=$(python3 scripts/vllm_metrics.py)
  python3 scripts/run_bench.py --config configs/selfhost/$1.toml --bench $B --n $3 --concurrency $4 --name $2
  after=$(python3 scripts/vllm_metrics.py)
  python3 -c "import json,sys; b,a=json.loads(sys.argv[1]),json.loads(sys.argv[2]); d={k:a[k]-b[k] for k in a}; d['hit_rate']=d['prefix_cache_hits']/max(1,d['prefix_cache_queries']); json.dump(d,open('results/$2/vllm_metrics.json','w'),indent=1); print('vllm', d)" "$before" "$after"
}
run qwen_full          exp_qwen_full          30 6
run qwen_nocache_full  exp_qwen_nocache_full  30 6
run qwen_truncate      exp_qwen_truncate      30 6
run qwen_checkpoint8k  exp_qwen_checkpoint8k  30 6
run qwen_full          exp_qwen_full_c1       10 1
run qwen_full          exp_qwen_full_c16      30 16
echo EXPERIMENT_DONE
