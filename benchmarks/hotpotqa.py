"""HotpotQA (distractor setting), as an agentic retrieval task.

Each question comes with 10 Wikipedia paragraphs, only 2 of which contain the supporting facts. The agent
does not see the paragraphs; it must use `search` (BM25 over the 10 paragraphs) and `read` (full
paragraph by title) to find evidence across both, then answer. Scored with the official HotpotQA answer
normalization: exact match decides `correct`; token F1 is the score.

Data: the official dev set (7,405 questions). The original CMU host is often unreachable, so questions are
fetched from the Hugging Face datasets-server JSON API (no extra libraries) and cached in data/. A fixed pool
of the first POOL dev questions is cached; runs sample from that pool with a seed.
"""
from __future__ import annotations

import collections
import json
import math
import re
import string

from harness.tools import Tool, ToolRegistry

from .base import DATA_DIR, Benchmark, Task, read_jsonl, sample

API = ("https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor"
       "&split=validation&offset={offset}&length=100")
POOL = 1500


def load_pool(pool: int = POOL) -> list[dict]:
    """Official-format records ({_id, question, answer, context: [[title, [sentences]]]}), cached."""
    import urllib.request
    path = DATA_DIR / f"hotpot_dev_distractor_first{pool}.jsonl"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        rows = []
        for offset in range(0, pool, 100):
            print(f"downloading HotpotQA dev rows {offset}-{offset + 99}")
            with urllib.request.urlopen(API.format(offset=offset), timeout=120) as r:
                for item in json.loads(r.read())["rows"]:
                    x = item["row"]
                    rows.append({"_id": x["id"], "question": x["question"], "answer": x["answer"],
                                 "type": x["type"], "level": x["level"],
                                 "context": [[t, s] for t, s in zip(x["context"]["title"], x["context"]["sentences"])]})
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    return read_jsonl(path)


# ---- official answer normalization and scoring ----------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = collections.Counter(pt) & collections.Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


# ---- BM25 over the question's own paragraphs ------------------------------------------------
def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = collections.Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _tok(query)
        out = []
        for d in self.docs:
            tf = collections.Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


def make_tools(context: list) -> ToolRegistry:
    titles = [t for t, _ in context]
    paras = {t: " ".join(sents) for t, sents in context}
    bm25 = BM25([f"{t} {paras[t]}" for t in titles])

    def search(query: str, k: int = 3) -> str:
        sc = bm25.scores(query)
        top = sorted(range(len(titles)), key=lambda i: -sc[i])[:k]
        return json.dumps([{"title": titles[i], "preview": context[i][1][0][:200] if context[i][1] else ""}
                           for i in top], ensure_ascii=False)

    def read(title: str) -> str:
        if title in paras:
            return paras[title]
        close = [t for t in titles if title.lower() in t.lower() or t.lower() in title.lower()]
        if close:
            return paras[close[0]]
        return f"No paragraph titled '{title}'. Known titles come from search results."

    return ToolRegistry([
        Tool("search", "Search the Wikipedia paragraphs for this question. Returns titles and previews.",
             {"type": "object", "properties": {"query": {"type": "string", "minLength": 1},
                                               "k": {"type": "integer", "minimum": 1, "maximum": 10}},
              "required": ["query"], "additionalProperties": False}, search),
        Tool("read", "Read the full paragraph with this exact title.",
             {"type": "object", "properties": {"title": {"type": "string", "minLength": 1}},
              "required": ["title"], "additionalProperties": False}, read),
    ])


PROMPT = ("Answer the question. The relevant facts are spread across several Wikipedia paragraphs that you "
          "can search and read. Answer with a short span (an entity, date, number, or yes/no), not a "
          "sentence.\n\nQuestion: {q}")


class HotpotQA(Benchmark):
    name = "hotpotqa"

    def load(self, n: int, seed: int = 0) -> list[Task]:
        data = load_pool()
        tasks = []
        for ex in sample(data, n, seed):
            gold, ctx = ex["answer"], ex["context"]
            tasks.append(Task(
                id=ex["_id"], prompt=PROMPT.format(q=ex["question"]), gold=gold,
                make_tools=lambda ctx=ctx: make_tools(ctx),
                score=lambda a, gold=gold: (f1_score(a or "", gold), exact_match(a or "", gold)),
                answer_check=lambda a: 0 < len(a.split()) <= 12,
            ))
        return tasks
