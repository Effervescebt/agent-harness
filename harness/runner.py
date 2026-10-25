"""Run a benchmark under one configuration and write traces plus a summary.

Output: results/<run name>/
  config.json     the exact config used (models, agent settings, benchmark, n, seed)
  traces.jsonl    one line per task: answer, gold, score, correct, full trajectory (or cascade attempts)
  summary.json    metrics.summarize(...) over all tasks
"""
from __future__ import annotations

import json
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path

from .agent import Agent, AgentConfig
from .llm import make_client
from .metrics import summarize
from .router import StepEscalation, needs_escalation

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    models = tomllib.loads((ROOT / "configs" / "models.toml").read_text())
    cfg["_models"] = models.get("models", {})
    return cfg


def model_spec(cfg: dict, key: str) -> dict:
    """A model reference in a run config is a name from configs/models.toml."""
    name = cfg["routing"][key]
    if name not in cfg["_models"]:
        raise SystemExit(f"model '{name}' is not defined in configs/models.toml")
    spec = {"name": name, **cfg["_models"][name]}
    if "cache" in cfg:  # [cache] enabled = true|false in the run config (Anthropic markers; others cache automatically)
        spec["cache"] = cfg["cache"].get("enabled", False)
    return spec


def prices(cfg: dict) -> dict:
    return {spec["model"]: {k: spec[k] for k in ("input", "output", "cached_input") if k in spec}
            for spec in cfg["_models"].values() if "input" in spec}


def agent_config(cfg: dict) -> AgentConfig:
    known = {f.name for f in fields(AgentConfig)}
    return AgentConfig(**{k: v for k, v in cfg.get("agent", {}).items() if k in known})


def run_task(task, cfg: dict) -> dict:
    acfg = agent_config(cfg)
    routing = cfg.get("routing", {})
    strategy = routing.get("strategy", "single")
    primary = make_client(model_spec(cfg, "primary"))
    summarizer = make_client(model_spec(cfg, "summarizer")) if routing.get("summarizer") else primary

    if strategy == "step_escalation":
        strong = make_client(model_spec(cfg, "strong"))
        router = StepEscalation(primary, strong, routing.get("escalate_after_repairs", 1),
                                routing.get("escalate_after_steps", 4))
        trajs = [Agent(primary, task.make_tools(), acfg, summarizer, router).run(task.id, task.prompt)]
    else:
        trajs = [Agent(primary, task.make_tools(), acfg, summarizer).run(task.id, task.prompt)]
        if strategy == "cascade" and needs_escalation(trajs[0], task.answer_check):
            strong = make_client(model_spec(cfg, "strong"))
            trajs.append(Agent(strong, task.make_tools(), acfg, summarizer).run(task.id, task.prompt))

    answer = trajs[-1].answer
    rec_extra = {}
    if getattr(task, "score_trajectory", None):  # session benchmarks score the submitted answers
        res = task.score_trajectory(trajs[-1].to_dict())
        score, correct, rec_extra = res["score"], res["score"] == 1.0, {"session": res}
    else:
        score, correct = task.score(answer)
    rec = {"task_id": task.id, "answer": answer, "gold": task.gold, "score": score, "correct": correct, **rec_extra}
    if len(trajs) > 1 or strategy == "cascade":
        rec["attempts"] = [t.to_dict() for t in trajs]
    else:
        rec["trajectory"] = trajs[0].to_dict()
    return rec


def run(config_path: str, bench_name: str, n: int, seed: int = 0, concurrency: int = 4,
        name: str | None = None, overrides: dict | None = None) -> Path:
    from benchmarks import BENCHMARKS

    cfg = load_config(config_path)
    for k, v in (overrides or {}).items():  # e.g. {"routing.primary": "mock"}
        sec, key = k.split(".", 1)
        cfg.setdefault(sec, {})[key] = v
    tasks = BENCHMARKS[bench_name]().load(n, seed)
    run_name = name or f"{time.strftime('%Y%m%d-%H%M%S')}_{bench_name}_{Path(config_path).stem}"
    # Provider caches are shared per account: tag the system prompt with the run name so one run never
    # reads cache entries written by another (constant within the run, so sessions still share it).
    from .agent import AgentConfig as _AC
    cfg.setdefault("agent", {})["system_prompt"] = (cfg["agent"].get("system_prompt") or _AC.system_prompt) \
        + f"\n[run: {run_name}]"
    out = ROOT / "results" / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {k: v for k, v in cfg.items() if k != "_models"} | {"benchmark": bench_name, "n": n, "seed": seed,
                                                            "models": cfg["_models"]}, indent=1))
    records, t_run = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex, open(out / "traces.jsonl", "w") as f:
        futs = {ex.submit(run_task, t, cfg): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(tasks):
                ok = sum(r["correct"] for r in records)
                print(f"  {i}/{len(tasks)} done, {ok} correct")
    summary = summarize(records, prices(cfg))
    summary["run_wall_s"] = time.perf_counter() - t_run
    summary["concurrency"] = concurrency
    gpu = cfg.get("gpu")  # self-hosted: bill the GPU-hours this run used
    if gpu:
        summary["gpu_cost_per_task_usd"] = gpu["usd_per_hour"] * summary["run_wall_s"] / 3600 / len(records)
        summary["sessions_per_gpu_hour"] = len(records) / (summary["run_wall_s"] / 3600)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "success_rate", "cost_per_task_usd", "latency_p50_s",
                                              "latency_p95_s", "mean_steps")}, indent=1))
    print(f"results: {out}")
    return out
