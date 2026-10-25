"""Read vLLM's prefix-cache counters from /metrics (server-side KV-cache hit accounting).

  python scripts/vllm_metrics.py            -> prints {"queries": ..., "hits": ...}
"""
import json
import os
import re
import urllib.request

base = os.environ["VLLM_BASE_URL"].rsplit("/v1", 1)[0]
req = urllib.request.Request(base + "/metrics", headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}",
                                                "User-Agent": "agent-harness/0.1"})
text = urllib.request.urlopen(req, timeout=30).read().decode()
out = {}
for key in ("prefix_cache_queries", "prefix_cache_hits", "prompt_tokens", "generation_tokens"):
    vals = [float(v) for v in re.findall(rf"^vllm:{key}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)$", text, re.M)]
    out[key] = sum(vals)
print(json.dumps(out))
