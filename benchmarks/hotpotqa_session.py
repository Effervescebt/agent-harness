"""HotpotQA-Session: a long-horizon agent workload built from HotpotQA.

One task is a session of G questions answered in a single conversation, the way a long-running agent
works through a queue. Search covers the union of all G questions' paragraphs (G x 10, so each question
also has the others' paragraphs as distractors). The agent submits each answer with `submit_answer`
and ends with `final_answer`. Sessions run 15-25 steps and grow to tens of thousands of prompt tokens,
which is where context management and prompt caching matter.

Score: mean exact match over the session's questions (official HotpotQA normalization); F1 is recorded.
"""
from __future__ import annotations

import json

from harness.tools import Tool, ToolRegistry

from .base import Benchmark, Task, sample
from .hotpotqa import BM25, exact_match, f1_score, load_pool

GROUP = 6

PROMPT = """Work through the {g} questions below, one at a time. For each question, search and read the
Wikipedia paragraphs to find the evidence (facts are often spread over two paragraphs), then call
submit_answer with the question number and a short answer span (an entity, date, number, or yes/no).
After all {g} answers are submitted, call final_answer with "done".

{questions}"""


def make_tools(contexts: list[list]) -> ToolRegistry:
    seen, titles, paras, first = set(), [], {}, {}
    for ctx in contexts:
        for t, sents in ctx:
            if t not in seen:
                seen.add(t)
                titles.append(t)
                paras[t] = " ".join(sents)
                first[t] = sents[0][:200] if sents else ""
    bm25 = BM25([f"{t} {paras[t]}" for t in titles])
    submitted: dict[int, str] = {}

    def search(query: str, k: int = 4) -> str:
        sc = bm25.scores(query)
        top = sorted(range(len(titles)), key=lambda i: -sc[i])[:k]
        return json.dumps([{"title": titles[i], "preview": first[titles[i]]} for i in top], ensure_ascii=False)

    def read(title: str) -> str:
        if title in paras:
            return paras[title]
        close = [t for t in titles if title.lower() in t.lower() or t.lower() in title.lower()]
        return paras[close[0]] if close else f"No paragraph titled '{title}'."

    def submit_answer(question: int, answer: str) -> str:
        submitted[question] = answer
        return f"Recorded answer for question {question}."

    return ToolRegistry([
        Tool("search", "Search all Wikipedia paragraphs in this session. Returns titles and previews.",
             {"type": "object", "properties": {"query": {"type": "string", "minLength": 1},
                                               "k": {"type": "integer", "minimum": 1, "maximum": 10}},
              "required": ["query"], "additionalProperties": False}, search),
        Tool("read", "Read the full paragraph with this exact title.",
             {"type": "object", "properties": {"title": {"type": "string", "minLength": 1}},
              "required": ["title"], "additionalProperties": False}, read),
        Tool("submit_answer", "Submit the answer to one question (by its number). Resubmitting overwrites.",
             {"type": "object", "properties": {"question": {"type": "integer", "minimum": 1},
                                               "answer": {"type": "string", "minLength": 1}},
              "required": ["question", "answer"], "additionalProperties": False}, submit_answer,
             side_effect_free=False),
    ])


def score_session(traj: dict, golds: list[str]) -> dict:
    """Read the submitted answers back from the trajectory (last submission per question wins)."""
    answers: dict[int, str] = {}
    for s in traj["steps"]:
        for c in s["tool_calls"]:
            if c["name"] == "submit_answer" and c["ok"] and c["args"]:
                answers[int(c["args"]["question"])] = c["args"]["answer"]
    em = [float(exact_match(answers.get(i + 1, ""), g)) for i, g in enumerate(golds)]
    f1 = [f1_score(answers.get(i + 1, ""), g) for i, g in enumerate(golds)]
    return {"answers": answers, "em": em, "f1": f1, "score": sum(em) / len(em)}


class HotpotQASession(Benchmark):
    name = "hotpotqa_session"

    def __init__(self, group: int = GROUP):
        self.group = group

    def load(self, n: int, seed: int = 0) -> list[Task]:
        items = sample(load_pool(), n * self.group, seed)
        tasks = []
        for s in range(0, len(items) - self.group + 1, self.group):
            grp = items[s:s + self.group]
            golds = [x["answer"] for x in grp]
            ctxs = [x["context"] for x in grp]
            qs = "\n".join(f"{i + 1}. {x['question']}" for i, x in enumerate(grp))
            task = Task(id="session-" + "-".join(x["_id"][:6] for x in grp),
                        prompt=PROMPT.format(g=self.group, questions=qs), gold=json.dumps(golds),
                        make_tools=lambda ctxs=ctxs: make_tools(ctxs),
                        score=lambda a: (0.0, False))
            task.score_trajectory = lambda traj, golds=golds: score_session(traj, golds)
            tasks.append(task)
        return tasks
