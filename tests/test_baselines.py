"""
Offline tests for the baseline harness: a scripted fake model stands in for a
real one, so no network or API key is needed.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import BadRequestError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.providers import InfraError, ModelSpec, OpenAICompatibleChat, QuotaExhausted
from inference import parse_model_response
from baselines.run import run_episode, run_model
from baselines.summarize import model_table, outcome, task_table
from environment.db_engine import DatabaseEngine
from environment.tasks import TASKS, get_task_by_id, resolve_ground_truth

SPEC = ModelSpec("fake", "fake", "fake-1", "a scripted fake")
USAGE = {"served_model": "fake-1", "tokens_in": 10, "tokens_out": 2}


def truth(task_id):
    db = DatabaseEngine()
    db.initialize()
    try:
        return resolve_ground_truth(get_task_by_id(task_id), db)
    finally:
        db.close()


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)

    def reply(self, messages):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return (reply if isinstance(reply, str) else json.dumps(reply)), USAGE


def honest(task_id):
    return [{"sql_query": get_task_by_id(task_id).ground_truth_sql},
            {"submit_answer": str(truth(task_id))}]


class TestEpisode(unittest.TestCase):
    def test_an_honest_episode_is_recorded(self):
        record = run_episode(FakeChat(honest("easy_user_count")), "easy_user_count")
        self.assertTrue(record["success"])
        self.assertTrue(record["verified"])
        self.assertEqual((record["turns"], record["env_steps"], record["parse_errors"]), (2, 2, 0))
        self.assertEqual(record["tokens_in"], 20)
        self.assertEqual(outcome(record), "correct")

    def test_a_malformed_reply_costs_a_turn_but_not_a_step(self):
        replies = ["sure, let me look"] + honest("easy_user_count")
        record = run_episode(FakeChat(replies), "easy_user_count")
        self.assertEqual((record["turns"], record["env_steps"], record["parse_errors"]), (3, 2, 1))
        self.assertTrue(record["success"])

    def test_answering_without_a_query_is_flagged(self):
        record = run_episode(FakeChat([{"submit_answer": str(truth("easy_user_count"))}]),
                             "easy_user_count")
        self.assertEqual(outcome(record), "correct_unchecked")

    def test_never_answering_is_its_own_outcome(self):
        record = run_episode(FakeChat([{"sql_query": "SELECT 1"}] * 15), "easy_user_count")
        self.assertIsNone(record["submitted"])
        self.assertEqual(outcome(record), "no_answer")


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def run_with(self, make_replies, tasks=("easy_user_count",), repeats=1):
        return run_model(SPEC, list(tasks), repeats, results_dir=self.dir,
                         chat_factory=lambda spec: FakeChat(make_replies()), log=lambda line: None)

    def test_finished_episodes_are_not_run_again(self):
        self.assertEqual(self.run_with(lambda: honest("easy_user_count"), repeats=2), 2)
        self.assertEqual(self.run_with(lambda: honest("easy_user_count"), repeats=2), 0)
        self.assertEqual(len((self.dir / "fake.jsonl").read_text().splitlines()), 2)

    def test_an_exhausted_quota_stops_the_model_without_scoring_it(self):
        recorded = self.run_with(lambda: [QuotaExhausted("daily limit")],
                                 tasks=("easy_user_count", "hard_top_spender"))
        self.assertEqual(recorded, 0)
        self.assertFalse((self.dir / "fake.jsonl").exists())

    def test_an_infrastructure_error_skips_only_that_episode(self):
        replies = iter([[InfraError("timeout")], honest("hard_top_spender")])
        recorded = self.run_with(lambda: next(replies), tasks=("easy_user_count", "hard_top_spender"))
        self.assertEqual(recorded, 1)
        record = json.loads((self.dir / "fake.jsonl").read_text())
        self.assertEqual(record["task_id"], "hard_top_spender")


class TestProviders(unittest.TestCase):
    def test_a_rejected_tool_call_is_the_models_reply_not_an_outage(self):
        """
        Found in the first full run: Groq rejects a generation in which the
        model calls a tool it was never offered (HTTP 400 tool_use_failed). The
        runner treated that as infrastructure and skipped the episode, which
        hid a model failure. It must come back as the model's (malformed) reply.
        """
        error_body = {"message": "Tool choice is none, but model called a tool",
                      "type": "invalid_request_error", "code": "tool_use_failed",
                      "failed_generation": '{"name": "assistant", "arguments": SELECT 1}'}
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"),
                                  json={"error": error_body})
        error = BadRequestError("tool_use_failed", response=response, body=error_body)

        def create(**kwargs):
            raise error

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        chat = OpenAICompatibleChat(ModelSpec("fake", "groq", "fake-1", "test"), client=client)
        text, usage = chat.reply([{"role": "user", "content": "hi"}])
        self.assertEqual(text, error_body["failed_generation"])
        self.assertIsNone(parse_model_response(text), "it should count as a parse error")


class TestSummary(unittest.TestCase):
    def record(self, task_id, success, verified=True, repeat=1):
        return {"model": "fake", "task_id": task_id, "repeat": repeat, "success": success,
                "verified": verified, "destructive": False, "env_steps": 2,
                "submitted": "x", "final_score": 0.94 if success else 0.01,
                "served_via": "a scripted fake"}

    def test_tables_count_what_happened(self):
        records = [self.record("easy_user_count", True, repeat=1),
                   self.record("easy_user_count", False, repeat=2),
                   self.record("easy_user_count", True, verified=False, repeat=3)]
        self.assertIn(f"| `fake` (partial: 3 of {3 * len(TASKS)} episodes) | 2/3 (67%)", model_table(records))
        self.assertIn("| `easy_user_count` | 2/3 |", task_table(records))

    def test_an_unfinished_run_is_labelled_partial(self):
        full = [dict(self.record(task.task_id, True), model="full") for task in TASKS]
        partial = [dict(self.record("easy_user_count", True), model="half")]
        table = model_table(full + partial)
        self.assertIn(f"`half` (partial: 1 of {len(TASKS)} episodes)", table)
        self.assertNotIn("`full` (partial", table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
