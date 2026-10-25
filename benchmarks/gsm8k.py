"""GSM8K (test split, 1,319 grade-school math word problems) with a calculator tool.

The agent can evaluate arithmetic with `calculator` (a safe AST evaluator, no eval()). Correct when the
final number matches the reference answer.
"""
from __future__ import annotations

import ast
import operator
import re

from harness.tools import Tool, ToolRegistry

from .base import Benchmark, Task, download, read_jsonl, sample

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")
    return ev(ast.parse(expr.replace(",", "").replace("^", "**"), mode="eval"))


def last_number(s: str) -> float | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def make_tools() -> ToolRegistry:
    def calculator(expression: str) -> str:
        v = safe_eval(expression)
        return str(int(v)) if float(v).is_integer() else f"{v:.6g}"

    return ToolRegistry([Tool(
        "calculator", "Evaluate an arithmetic expression, e.g. '(12 * 3 + 4) / 2'.",
        {"type": "object", "properties": {"expression": {"type": "string", "minLength": 1}},
         "required": ["expression"], "additionalProperties": False}, calculator)])


PROMPT = ("Solve the math word problem. Use the calculator for arithmetic. Give the final answer as a "
          "single number in final_answer.\n\nProblem: {q}")


class GSM8K(Benchmark):
    name = "gsm8k"

    def load(self, n: int, seed: int = 0) -> list[Task]:
        rows = read_jsonl(download(URL, "gsm8k_test.jsonl"))
        tasks = []
        for i, ex in enumerate(sample(list(enumerate(rows)), n, seed)):
            idx, row = ex
            gold = row["answer"].split("####")[-1].strip()
            g = last_number(gold)

            def score(a, g=g):
                p = last_number(a or "")
                ok = p is not None and g is not None and abs(p - g) < 1e-6
                return float(ok), ok

            tasks.append(Task(id=f"gsm8k-{idx}", prompt=PROMPT.format(q=row["question"]), gold=gold,
                              make_tools=make_tools, score=score,
                              answer_check=lambda a: last_number(a) is not None))
        return tasks
