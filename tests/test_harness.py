"""Unit tests (offline, mock model only):  python -m unittest discover tests"""
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.gsm8k import last_number, safe_eval
from benchmarks.gsm8k import make_tools as gsm_tools
from benchmarks.hotpotqa import exact_match, f1_score
from benchmarks.hotpotqa import make_tools as hotpot_tools
from harness.agent import Agent, AgentConfig, parse_text_tool_calls
from harness.context import truncate_old_tool_results
from harness.llm import Completion, MockClient, ToolCall, Usage
from harness.metrics import call_cost, summarize
from harness.router import StepEscalation, needs_escalation
from harness.tools import Tool, ToolRegistry, validate


def call(name, args, raw=None):
    return Completion("", [ToolCall(f"id_{name}_{time.perf_counter_ns()}", name, raw if raw is not None else json.dumps(args))],
                      Usage(100, 10), 0.01, "mock")


ECHO = Tool("echo", "echo", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
                             "additionalProperties": False}, lambda x: f"echo:{x}")


class TestValidation(unittest.TestCase):
    def test_schema_errors(self):
        s = {"type": "object", "properties": {"k": {"type": "integer", "minimum": 1}, "m": {"enum": ["a", "b"]}},
             "required": ["k"], "additionalProperties": False}
        self.assertEqual(validate({"k": 2, "m": "a"}, s), [])
        self.assertTrue(any("missing" in e for e in validate({}, s)))
        self.assertTrue(any("expected integer" in e for e in validate({"k": "2"}, s)))
        self.assertTrue(any("one of" in e for e in validate({"k": 1, "m": "z"}, s)))
        self.assertTrue(any("unexpected" in e for e in validate({"k": 1, "zz": 0}, s)))
        self.assertTrue(validate({"k": True}, s))  # bool is not an integer here

    def test_parse_errors(self):
        reg = ToolRegistry([ECHO])
        self.assertEqual(reg.parse("nope", "{}")[1].error_kind, "unknown_tool")
        self.assertEqual(reg.parse("echo", "{bad json")[1].error_kind, "parse")
        self.assertEqual(reg.parse("echo", '{"x": 3}')[1].error_kind, "schema")
        self.assertEqual(reg.parse("echo", '{"x": "hi"}')[0], {"x": "hi"})


class TestAgentLoop(unittest.TestCase):
    def test_recovers_from_malformed_call(self):
        client = MockClient([call("echo", None, raw="{not json"), call("echo", {"x": "a"}),
                             call("final_answer", {"answer": "done"})])
        traj = Agent(client, ToolRegistry([ECHO]), AgentConfig(max_steps=5)).run("t", "go")
        self.assertEqual((traj.status, traj.answer), ("answered", "done"))
        self.assertTrue(traj.steps[0].repaired)
        self.assertEqual(traj.steps[0].tool_calls[0]["error_kind"], "parse")

    def test_gives_up_after_repairs(self):
        client = MockClient(["just text"] * 10)
        traj = Agent(client, ToolRegistry([ECHO]), AgentConfig(max_steps=10, max_repairs=2)).run("t", "go")
        self.assertEqual(traj.status, "repairs_exhausted")
        self.assertEqual(len(traj.steps), 3)

    def test_forced_final_answer_on_last_step(self):
        seen_tools = []

        def script(messages, tools):
            seen_tools.append(sorted(t["name"] for t in tools))
            if tools == [t for t in tools if t["name"] == "final_answer"]:
                return call("final_answer", {"answer": "best guess"})
            return call("echo", {"x": "again"})

        traj = Agent(MockClient(script), ToolRegistry([ECHO]), AgentConfig(max_steps=3)).run("t", "go")
        self.assertEqual((traj.status, traj.answer), ("answered", "best guess"))
        self.assertEqual(seen_tools[-1], ["final_answer"])

    def test_parallel_tools_are_faster(self):
        slow = Tool("slow", "sleep", {"type": "object", "properties": {"i": {"type": "integer"}}, "required": ["i"]},
                    lambda i: time.sleep(0.25) or str(i))
        two = Completion("", [ToolCall("a", "slow", '{"i": 1}'), ToolCall("b", "slow", '{"i": 2}'),
                              ToolCall("c", "slow", '{"i": 3}')], Usage(1, 1), 0.0, "mock")
        timings = {}
        for par in (False, True):
            client = MockClient([two, call("final_answer", {"answer": "ok"})])
            traj = Agent(client, ToolRegistry([slow]), AgentConfig(parallel_tools=par)).run("t", "go")
            timings[par] = traj.wall_s
            self.assertEqual(traj.status, "answered")
        self.assertLess(timings[True], 0.5)
        self.assertGreater(timings[False], 0.7)

    def test_text_tool_mode(self):
        calls = parse_text_tool_calls('Sure.\n```json\n{"tool_calls": [{"name": "echo", "arguments": {"x": "q"}}]}\n```')
        self.assertEqual((calls[0].name, json.loads(calls[0].arguments)), ("echo", {"x": "q"}))
        self.assertIsNone(parse_text_tool_calls("no json here"))
        client = MockClient(['{"tool_calls": [{"name": "echo", "arguments": {"x": "q"}}]}',
                             '{"tool_calls": [{"name": "final_answer", "arguments": {"answer": "z"}}]}'])
        traj = Agent(client, ToolRegistry([ECHO]), AgentConfig(text_tool_mode=True)).run("t", "go")
        self.assertEqual(traj.answer, "z")


class TestContext(unittest.TestCase):
    def test_truncate_keeps_recent(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "tool", "tool_call_id": str(i), "content": "x" * 1000} for i in range(4)]
        out = truncate_old_tool_results(msgs, keep_last=2, old_chars=50)
        self.assertLess(len(out[1]["content"]), 200)
        self.assertEqual(len(out[-1]["content"]), 1000)
        self.assertEqual(len(msgs[1]["content"]), 1000)  # original untouched


class TestCacheAwareCompaction(unittest.TestCase):
    """Rolling truncation rewrites history every step (cache-breaking); checkpoint compaction rewrites it
    only at batched compaction events and is append-only in between."""

    def run_with(self, compression):
        big = Tool("big", "big output", {"type": "object", "properties": {"i": {"type": "integer"}}, "required": ["i"]},
                   lambda i: f"r{i}:" + "x" * 900)
        script = [call("big", {"i": i}) for i in range(8)] + [call("final_answer", {"answer": "ok"})]
        cfg = AgentConfig(max_steps=12, compression=compression, keep_last=2, old_chars=100, checkpoint_chars=2500)
        return Agent(MockClient(script), ToolRegistry([big]), cfg).run("t", "go")

    def test_prefix_breaks(self):
        none, rolling, ckpt = (self.run_with(c) for c in ("none", "truncate", "checkpoint"))
        self.assertEqual(none.prefix_breaks, 0)
        self.assertGreaterEqual(rolling.prefix_breaks, 6)          # one break per step once results age out
        self.assertGreater(ckpt.compactions, 0)
        self.assertEqual(ckpt.prefix_breaks, ckpt.compactions)     # breaks only at compaction events
        self.assertLess(ckpt.prefix_breaks, rolling.prefix_breaks)
        tokens = lambda t: sum(s.prompt_tokens_est for s in t.steps)
        self.assertLess(tokens(ckpt), tokens(none))                # still saves tokens


class TestMetricsAndRouting(unittest.TestCase):
    def test_cost(self):
        prices = {"m": {"input": 1.0, "cached_input": 0.1, "output": 4.0}}
        c = call_cost("m", {"input_tokens": 1_000_000, "cached_input_tokens": 500_000, "output_tokens": 250_000}, prices)
        self.assertAlmostEqual(c, 0.5 + 0.05 + 1.0)
        # cache writes priced at the write rate; input_tokens is the total prompt
        p2 = {"m": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 0.0}}
        c2 = call_cost("m", {"input_tokens": 3_000_000, "cached_input_tokens": 1_000_000,
                             "cache_write_tokens": 1_000_000, "output_tokens": 0}, p2)
        self.assertAlmostEqual(c2, 1.0 + 0.1 + 1.25)

    def test_summarize(self):
        step = {"model": "m", "usage": {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 0},
                "repaired": False, "tool_calls": [{"ok": True}, {"ok": False}]}
        recs = [{"trajectory": {"steps": [step], "wall_s": 1.0, "status": "answered"}, "correct": True, "score": 1.0},
                {"attempts": [{"steps": [step], "wall_s": 1.0, "status": "max_steps"},
                              {"steps": [step], "wall_s": 2.0, "status": "answered"}], "correct": False, "score": 0.0}]
        s = summarize(recs, {"m": {"input": 1.0, "output": 1.0}})
        self.assertEqual(s["success_rate"], 0.5)
        self.assertEqual(s["escalation_rate"], 0.5)
        self.assertAlmostEqual(s["tool_error_rate"], 0.5)

    def test_router(self):
        class T:  # minimal trajectory stand-in
            def __init__(self, status, answer, repaired=0):
                self.status, self.answer = status, answer
                self.steps = [type("S", (), {"repaired": True})()] * repaired
        cheap, strong = object(), object()
        r = StepEscalation(cheap, strong, escalate_after_repairs=1, escalate_after_steps=4)
        self.assertIs(r(0, T("running", None)), cheap)
        self.assertIs(r(1, T("running", None, repaired=1)), strong)
        self.assertIs(r(4, T("running", None)), strong)
        self.assertTrue(needs_escalation(T("max_steps", None)))
        self.assertFalse(needs_escalation(T("answered", "42"), lambda a: a.isdigit()))
        self.assertTrue(needs_escalation(T("answered", "forty"), lambda a: a.isdigit()))


class TestBenchmarks(unittest.TestCase):
    def test_hotpot_scoring_and_tools(self):
        self.assertTrue(exact_match("The Eiffel Tower", "eiffel tower"))
        self.assertAlmostEqual(f1_score("Paris France", "Paris"), 2 / 3)
        self.assertEqual(f1_score("yes", "no"), 0.0)
        reg = hotpot_tools([["Paris", ["Paris is the capital of France.", "It has the Louvre."]],
                            ["Berlin", ["Berlin is the capital of Germany."]]])
        hits = json.loads(reg.run("search", {"query": "capital of France"}).content)
        self.assertEqual(hits[0]["title"], "Paris")
        self.assertIn("Louvre", reg.run("read", {"title": "Paris"}).content)

    def test_gsm8k_tools(self):
        self.assertEqual(safe_eval("(12 * 3 + 4) / 2"), 20)
        self.assertEqual(safe_eval("2^10"), 1024)
        with self.assertRaises(ValueError):
            safe_eval("__import__('os')")
        self.assertEqual(last_number("so the answer is 1,234."), 1234.0)
        self.assertEqual(gsm_tools().run("calculator", {"expression": "7*6"}).content, "42")


if __name__ == "__main__":
    unittest.main()
