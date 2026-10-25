# Results

Runs from 2026-10-25. Per-run metrics are in `results_summary/`; prices are in `configs/models.toml`.

## Setup

Benchmark: HotpotQA-Session. Six HotpotQA questions per session, answered in one conversation, with BM25
search over the 60 paragraphs of those questions. Score is exact match per question. Claude Haiku 5.5 and
DeepSeek ran the same 30 seeded sessions (180 questions); Claude Sonnet 5.5 ran the first 20 (120
questions). Thinking was off on the Claude models. Prompt caching was on in every run except one Haiku
baseline. Each run tagged its system prompt with its own name, so no run read another run's cache.

Context strategies:

| Strategy | What it does to old tool results |
|---|---|
| Full history | Nothing. |
| Rolling truncation | Every step, cuts each result older than the last two to 200 characters. |
| Checkpoint compaction | Leaves them untouched until the uncompacted ones exceed 2,000 or 8,000 characters, then cuts all of them in one step. |

A prefix break is a step whose prompt is not the previous prompt with new messages appended. Every break
invalidates the provider's cache from the edited message onward.

## Hosted models

### Claude Haiku 5.5 (n = 30 sessions x 6 questions)

| Strategy | Question EM (95% CI) | $ / session | vs full | Cache hit | Prefix breaks / session | Prompt tokens / session | p50 latency (s) | Steps |
|---|---|---|---|---|---|---|---|---|
| Full history, caching off | 0.483 [0.394, 0.567] | 0.00520 | +152.4% | 0.0% | 0.0 | 41,797 | 10.5 | 7.9 |
| Full history | 0.506 [0.422, 0.589] | 0.00206 | +0.0% | 81.5% | 0.0 | 39,497 | 11.2 | 7.7 |
| Rolling truncation | 0.478 [0.406, 0.550] | 0.00455 | +121.0% | 26.1% | 7.2 | 42,702 | 15.7 | 9.4 |
| Checkpoint compaction (2K chars) | 0.494 [0.422, 0.567] | 0.00342 | +66.1% | 54.9% | 3.8 | 42,486 | 14.2 | 9.2 |
| Checkpoint compaction (8K chars) | 0.511 [0.433, 0.589] | 0.00247 | +20.1% | 75.7% | 0.9 | 42,342 | 13.3 | 8.6 |

### DeepSeek (deepseek-flash) (n = 30 sessions x 6 questions)

| Strategy | Question EM (95% CI) | $ / session | vs full | Cache hit | Prefix breaks / session | Prompt tokens / session | p50 latency (s) | Steps |
|---|---|---|---|---|---|---|---|---|
| Full history | 0.583 [0.494, 0.667] | 0.00357 | +0.0% | 85.6% | 0.0 | 55,900 | 19.2 | 8.8 |
| Rolling truncation | 0.489 [0.383, 0.589] | 0.00788 | +121.0% | 80.7% | 13.8 | 106,893 | 41.7 | 17.3 |
| Checkpoint compaction (2K chars) | 0.522 [0.422, 0.622] | 0.00935 | +162.0% | 83.1% | 11.4 | 143,483 | 57.5 | 19.1 |
| Checkpoint compaction (8K chars) | 0.522 [0.422, 0.622] | 0.00712 | +99.6% | 81.5% | 3.4 | 103,163 | 41.4 | 14.8 |

### Claude Sonnet 5.5 (n = 20 sessions x 6 questions)

| Strategy | Question EM (95% CI) | $ / session | vs full | Cache hit | Prefix breaks / session | Prompt tokens / session | p50 latency (s) | Steps |
|---|---|---|---|---|---|---|---|---|
| Full history | 0.617 [0.550, 0.692] | 0.03933 | +0.0% | 79.8% | 0.0 | 36,850 | 17.4 | 6.7 |
| Rolling truncation | 0.608 [0.517, 0.692] | 0.17631 | +348.3% | 16.3% | 10.4 | 83,217 | 32.1 | 12.7 |
| Checkpoint compaction (2K chars) | 0.625 [0.525, 0.725] | 0.13601 | +245.8% | 47.7% | 6.8 | 88,176 | 37.9 | 12.9 |
| Checkpoint compaction (8K chars) | 0.617 [0.550, 0.683] | 0.06697 | +70.3% | 74.0% | 1.4 | 62,347 | 22.6 | 9.8 |

### Prompt caching

On Haiku with full history, turning caching on cut cost per session from $0.00520 to $0.00206 (60%).
Accuracy moved from 0.483 to 0.506, inside the confidence intervals.

### Rolling truncation

Rolling truncation was the most expensive strategy on all three models: 2.2x the full-history cost on
Haiku and DeepSeek and 4.5x on Sonnet ($0.0393 to $0.1763 per session). Two things in the traces account
for it.

1. It broke the prefix 7.2 times per session on Haiku, 10.4 on Sonnet and 13.8 on DeepSeek. On Sonnet the
   cache hit rate fell from 79.8% to 16.3%, and the share of prompt tokens billed as cache writes, which
   cost 1.25x the input price, rose from 20.2% to 83.7%.
2. The agent re-searched and re-read paragraphs whose text had been cut. Steps per session went from 6.7 to
   12.7 on Sonnet and from 8.8 to 17.3 on DeepSeek, and total prompt tokens per session went up, not down:
   36.9K to 83.2K on Sonnet.

Accuracy did not improve on any model (Sonnet 0.617 to 0.608, Haiku 0.506 to 0.478, DeepSeek 0.583 to 0.489).

### Checkpoint compaction

Checkpoint compaction is the strategy this harness adds. It compacts the same tool results as rolling
truncation, but in batches, and leaves the prompt append-only between batches. In every run its prefix
breaks matched its compaction count (1.40 and 1.40 per session on Sonnet at 8,000 characters), so the
prompt was rewritten only at compaction steps.

Results at the 8,000-character threshold, against rolling truncation:

| | Sonnet | Haiku | DeepSeek |
|---|---|---|---|
| Prefix breaks per session | 10.4 -> 1.4 (-86%) | 7.2 -> 0.9 (-88%) | 13.8 -> 3.4 (-75%) |
| Cache hit rate | 16.3% -> 74.0% | 26.1% -> 75.7% | 80.7% -> 81.5% |
| Prompt tokens billed as cache writes | 83.7% -> 26.0% | 73.9% -> 24.3% | no write charge |
| Steps per session | 12.7 -> 9.8 | 9.4 -> 8.6 | 17.3 -> 14.8 |
| Cost per session | $0.1763 -> $0.0670 (-62%) | $0.00455 -> $0.00247 (-46%) | $0.00788 -> $0.00712 (-10%) |
| Question EM | 0.608 -> 0.617 | 0.478 -> 0.511 | 0.489 -> 0.522 |

The threshold controls the trade-off. At 2,000 characters Sonnet compacted 7.75 times per session, the
hit rate was 47.7% and cost per session was $0.1360, 23% below rolling truncation; at 8,000 characters it
compacted 1.4 times and cost 62% less.

DeepSeek gained the least because its cache bills no write premium and still hit 80.7% of prompt tokens
under rolling truncation; its extra cost came from extra steps.

Full history with caching was still the cheapest configuration on all three models: $0.0393 per session
on Sonnet against $0.0670 for checkpoint compaction.

### Limits

One seed, 120-180 questions per run. All accuracy differences on Claude fall inside the 95% intervals. On
DeepSeek, rolling truncation scored 0.489 against 0.583 for full history, with overlapping intervals.

## Self-hosted model

Qwen3-8B in BF16 with thinking off, served by vLLM on one NVIDIA L40 48GB at $0.69 per hour (RunPod
community cloud), on the same sessions. Cost per session is the GPU time each run used: price per hour
times wall time, divided by sessions. Model loading (7.5 minutes) is not included. The cache hit rate is
vLLM's own `prefix_cache_hits / prefix_cache_queries` over the run. The cache-off run sent a random
`cache_salt` with every request to the same server. GPU time for all runs cost about $0.31.

### Self-hosted Qwen3-8B on vLLM (one NVIDIA L40, $0.69/h), HotpotQA-Session

| Run | Sessions | Concurrency | Question EM (95% CI) | GPU $ / session | Server KV-cache hit | Sessions / GPU-hour | p50 session latency (s) | Steps |
|---|---|---|---|---|---|---|---|---|
| Full history, prefix cache on | 30 | 6 | 0.339 [0.267, 0.411] | 0.00083 | 91.7% | 827 | 21.8 | 19.1 |
| Full history, prefix cache off (salted) | 30 | 6 | 0.383 [0.317, 0.456] | 0.00220 | 0.0% | 313 | 58.8 | 18.8 |
| Rolling truncation, cache on | 30 | 6 | 0.372 [0.317, 0.433] | 0.00091 | 83.2% | 761 | 23.1 | 17.8 |
| Checkpoint compaction (8K), cache on | 30 | 6 | 0.333 [0.256, 0.400] | 0.00092 | 89.4% | 752 | 22.2 | 19.6 |
| Full history, cache on | 10 | 1 | 0.317 [0.200, 0.450] | 0.00530 | 91.1% | 130 | 20.5 | 21.2 |
| Full history, cache on | 30 | 16 | 0.339 [0.267, 0.417] | 0.00071 | 91.4% | 974 | 29.0 | 21.7 |

### Same benchmark through APIs (full history, caching on)

| Model | Question EM | $ / session |
|---|---|---|
| Claude Haiku 5.5 | 0.506 | 0.00206 |
| DeepSeek (deepseek-flash) | 0.583 | 0.00357 |
| Claude Sonnet 5.5 | 0.617 | 0.03933 |

### Prefix caching on vLLM

Over 30 sessions the server processed 1.64M prompt tokens and generated 17K, so 99% of the work was
prefill. With prefix caching, 91.7% of prompt tokens were served from the KV cache. Against the cache-off
run on the same GPU: throughput 313 -> 827 sessions per GPU-hour (2.6x), median session latency 58.8 s ->
21.8 s (-63%), GPU cost per session $0.00220 -> $0.00083 (-62%). Accuracy was 0.339 with the cache and
0.383 without, inside the intervals.

### Concurrency

With prefix caching on, raising concurrency from 1 to 16 sessions raised throughput from 130 to 974
sessions per GPU-hour (7.5x) and cut cost per session from $0.00530 to $0.00071 (-87%). Median session
latency rose from 20.5 s to 29.0 s.

### Against the APIs

At 16 concurrent sessions the self-hosted model cost $0.00071 per session, against $0.0393 for Claude
Sonnet 5.5 (55x more) and $0.00206 for Claude Haiku 5.5 (2.9x more). Its question EM was 0.339, against
0.617 and 0.506.

### Rolling truncation on vLLM

Rolling truncation broke the prefix 9.7 times per session, and vLLM still served 83.2% of prompt tokens
from cache, against 16.3% on the Claude API. Cost per session was $0.00091 against $0.00083 for full
history.

### Limits

One GPU type, one model, one seed, 30 sessions per run (10 for the single-session run). Cost assumes the
GPU was busy for the whole run.
