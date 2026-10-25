"""Benchmark interface. A benchmark yields Tasks; each Task brings its own tools and scorer."""
from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness.tools import ToolRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Task:
    id: str
    prompt: str
    gold: str
    make_tools: Callable[[], ToolRegistry]          # fresh tools per run (tools may hold per-task state)
    score: Callable[[str | None], tuple[float, bool]]  # answer -> (score in [0, 1], correct)
    answer_check: Callable[[str], bool] | None = None  # cheap validity check used by the cascade router


class Benchmark:
    name = "base"

    def load(self, n: int, seed: int = 0) -> list[Task]:
        raise NotImplementedError


def download(url: str, filename: str) -> Path:
    """Download once into data/ and reuse."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        print(f"downloading {url} -> {path}")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def sample(items: list, n: int, seed: int) -> list:
    """Deterministic subset so every config is evaluated on exactly the same tasks."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items[:n] if n else items


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
