#!/usr/bin/env python3
"""
Regression suite for the OpenEnv SQL Analyst environment.

Every test here pins a defect that was found by executing the environment
rather than by reading it. They are grouped by the property they protect:

  GroundTruth  - the graded answer must be derived from data, never declared
  ReadOnly     - the agent must not be able to mutate or reconfigure the DB
  Runtime      - the environment must behave identically in a worker thread
  Scoring      - documented reward/score behaviour must actually happen

Run:  python tests/test_environment.py
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import db_engine as db_engine_module
from environment.db_engine import DatabaseEngine, QueryStatus, MAX_FETCH_ROWS
from environment.env import (
    MAX_STEPS,
    SQLAnalystEnv,
    REWARD_SUCCESSFUL_QUERY,
    REWARD_SYNTAX_ERROR,
    REWARD_DESTRUCTIVE_ACTION,
    REWARD_INFINITE_LOOP,
)
from environment.models import Action
from environment.graders import calculate_final_score, grade_answer
from environment.tasks import TASKS, resolve_ground_truth, get_task_by_id


class TestGroundTruthIsDerived(unittest.TestCase):
    """
    Ground truth must come from each task's reference SQL, executed against
    the same database the agent queries.

    Regression: `medium_usa_revenue` shipped with a hardcoded ground truth of
    2423.87 while its own reference SQL returned 2204.87. A correct agent was
    graded wrong on every episode of that task.
    """

    def setUp(self):
        self.db = DatabaseEngine()
        self.db.initialize()

    def tearDown(self):
        self.db.close()

    def test_every_task_ground_truth_matches_its_reference_sql(self):
        for task in TASKS:
            with self.subTest(task=task.task_id):
                derived = resolve_ground_truth(task, self.db)
                self.assertIsNotNone(
                    derived,
                    f"{task.task_id}: reference SQL produced no answer",
                )

    def test_medium_task_returns_the_real_revenue(self):
        task = get_task_by_id("medium_usa_revenue")
        self.assertAlmostEqual(resolve_ground_truth(task, self.db), 2204.87, places=2)

    def test_easy_and_hard_ground_truths(self):
        self.assertEqual(resolve_ground_truth(get_task_by_id("easy_user_count"), self.db), 15)
        self.assertEqual(resolve_ground_truth(get_task_by_id("hard_top_spender"), self.db), "alice")

    def test_a_correct_answer_is_graded_correct(self):
        """The end-to-end property the 2423.87 bug violated."""
        env = SQLAnalystEnv()
        env.reset(task_id="medium_usa_revenue")
        _, _, done, info = env.step(Action(submit_answer="2204.87"))
        self.assertTrue(done)
        self.assertTrue(info["success"], "a correct answer must grade as correct")
        env.close()

    def test_ground_truth_cannot_be_declared_on_a_task(self):
        """Tasks must not carry a literal answer that could drift from data."""
        for task in TASKS:
            with self.subTest(task=task.task_id):
                self.assertFalse(
                    hasattr(task, "ground_truth"),
                    "Task must derive its answer, not store one",
                )


class TestReadOnlyEnforcement(unittest.TestCase):
    """
    The database must be read-only from the agent's perspective.

    Regression: the original regex denylist covered only six keywords. It
    missed REPLACE / CREATE / ATTACH (all of which actually mutated the
    database) and it fired on innocent SELECTs containing a keyword inside a
    string literal.
    """

    def setUp(self):
        self.db = DatabaseEngine()
        self.db.initialize()

    def tearDown(self):
        self.db.close()

    def test_classic_mutations_are_denied(self):
        for query in [
            "DELETE FROM users",
            "UPDATE users SET username='x'",
            "DROP TABLE users",
            "INSERT INTO users VALUES (99,'x','x','x','x')",
        ]:
            with self.subTest(query=query):
                _, status = self.db.execute_query(query)
                self.assertEqual(status, QueryStatus.DENIED)

    def test_bypasses_that_defeated_the_regex_are_denied(self):
        for query in [
            "REPLACE INTO users (user_id,username,email,country,created_at) "
            "VALUES (1,'PWNED','x','X','2020')",
            "CREATE TABLE evil AS SELECT * FROM users",
            "ATTACH DATABASE ':memory:' AS side",
            "PRAGMA query_only = OFF",
        ]:
            with self.subTest(query=query.split()[0]):
                _, status = self.db.execute_query(query)
                self.assertEqual(status, QueryStatus.DENIED)

    def test_data_is_unchanged_after_a_mutation_attempt(self):
        self.db.execute_query(
            "REPLACE INTO users (user_id,username,email,country,created_at) "
            "VALUES (1,'PWNED','x','X','2020')"
        )
        result, status = self.db.execute_query("SELECT username FROM users WHERE user_id=1")
        self.assertEqual(status, QueryStatus.OK)
        self.assertIn("alice", result)
        self.assertNotIn("PWNED", result)

    def test_keyword_inside_a_string_literal_is_not_a_mutation(self):
        """The false-positive direction: this SELECT must run, not end the episode."""
        result, status = self.db.execute_query(
            "SELECT COUNT(*) FROM users WHERE username = 'drop table'"
        )
        self.assertEqual(status, QueryStatus.OK, f"legit SELECT was rejected: {result}")

    def test_destructive_action_ends_the_episode(self):
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")
        _, reward, done, _ = env.step(Action(sql_query="DELETE FROM users"))
        self.assertEqual(reward.value, REWARD_DESTRUCTIVE_ACTION)
        self.assertTrue(done)
        env.close()


class TestRuntimeBehaviour(unittest.TestCase):
    """
    The environment must behave the same in a worker thread as in the main
    thread, because FastAPI runs synchronous endpoints in a threadpool.

    Regression: the timeout used signal.SIGALRM, which raises ValueError off
    the main thread. Every query issued through the HTTP server failed with
    "signal only works in main thread of the main interpreter".
    """

    def test_query_succeeds_in_a_worker_thread(self):
        outcome = {}

        def work():
            db = DatabaseEngine()
            db.initialize()
            outcome["result"] = db.execute_query("SELECT COUNT(*) FROM users")
            db.close()

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

        result, status = outcome["result"]
        self.assertEqual(status, QueryStatus.OK, f"query failed off-main-thread: {result}")
        self.assertIn("15", result)

    def test_timeout_interrupts_a_runaway_query_in_a_worker_thread(self):
        original = db_engine_module.QUERY_TIMEOUT
        db_engine_module.QUERY_TIMEOUT = 0.3
        outcome = {}

        def work():
            db = DatabaseEngine()
            db.initialize()
            # Cartesian product over 15 rows, 7 ways: ~170M rows.
            outcome["result"] = db.execute_query(
                "SELECT COUNT(*) FROM users a, users b, users c, users d, "
                "users e, users f, users g"
            )
            db.close()

        try:
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(timeout=20)
            result, status = outcome["result"]
            self.assertEqual(status, QueryStatus.ERROR)
            self.assertIn("timeout", result.lower())
        finally:
            db_engine_module.QUERY_TIMEOUT = original

    def test_oom_protection_caps_returned_rows(self):
        db = DatabaseEngine()
        db.initialize()
        result, status = db.execute_query("SELECT * FROM users a, users b")  # 225 rows
        self.assertEqual(status, QueryStatus.OK)
        data_rows = [ln for ln in result.splitlines() if ln.startswith("|") and "---" not in ln]
        self.assertLessEqual(len(data_rows) - 1, MAX_FETCH_ROWS)
        self.assertIn("TRUNCATED", result)
        db.close()


class TestScoring(unittest.TestCase):
    """
    Documented scoring behaviour must actually reach the final score.

    Regression: the grader computed 0.5 partial credit for a near-miss numeric
    answer, then env.py discarded it and returned the 0.01 floor, so a
    near-miss scored identically to a blank answer.
    """

    def test_partial_credit_survives_into_the_final_score(self):
        self.assertAlmostEqual(calculate_final_score(False, 0.5, 3), 0.5, places=2)

    def test_a_plain_wrong_answer_still_floors(self):
        self.assertAlmostEqual(calculate_final_score(False, 0.01, 3), 0.01, places=2)

    def test_correct_answers_earn_an_efficiency_bonus(self):
        fast = calculate_final_score(True, 0.99, 2)
        slow = calculate_final_score(True, 0.99, 12)
        self.assertGreater(fast, slow)

    def test_scores_stay_strictly_inside_zero_and_one(self):
        """OpenEnv requires scores in the open interval (0, 1)."""
        for correct, grade, steps in [
            (True, 0.99, 0), (True, 0.99, 15), (False, 0.0, 1), (False, 1.0, 1)
        ]:
            with self.subTest(correct=correct, steps=steps):
                score = calculate_final_score(correct, grade, steps)
                self.assertGreater(score, 0.0)
                self.assertLess(score, 1.0)

    def test_reward_shaping_matches_the_documented_table(self):
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")
        _, ok, _, _ = env.step(Action(sql_query="SELECT COUNT(*) FROM users"))
        self.assertEqual(ok.value, REWARD_SUCCESSFUL_QUERY)
        _, bad, _, _ = env.step(Action(sql_query="SELECT * FRM users"))
        self.assertEqual(bad.value, REWARD_SYNTAX_ERROR)
        env.close()


class TestGraderCannotBeGamed(unittest.TestCase):
    """
    A final answer is singular. Listing candidates must not score as correct.

    Regression: compare_values used unanchored substring containment, so an
    agent submitting every username scored 0.99 on the top-spender task.
    """

    def setUp(self):
        self.db = DatabaseEngine()
        self.db.initialize()
        self.domain = self.db.execute_privileged_column("SELECT username FROM users")

    def tearDown(self):
        self.db.close()

    def test_shotgunning_every_candidate_is_rejected(self):
        is_correct, score = grade_answer(
            "alice bob charlie diana eve frank grace", "alice", self.db, self.domain
        )
        self.assertFalse(is_correct, "listing every candidate must not score correct")
        self.assertLess(score, 0.5)

    def test_two_candidates_is_still_hedging(self):
        is_correct, _ = grade_answer("alice or karen", "alice", self.db, self.domain)
        self.assertFalse(is_correct)

    def test_a_verbose_single_answer_still_passes(self):
        is_correct, _ = grade_answer("The top spender is alice.", "alice", self.db, self.domain)
        self.assertTrue(is_correct, "a natural-language answer naming one value must pass")

    def test_exact_answer_passes(self):
        is_correct, _ = grade_answer("alice", "alice", self.db, self.domain)
        self.assertTrue(is_correct)

    def test_substring_of_a_longer_word_is_not_a_match(self):
        is_correct, _ = grade_answer("malice", "alice", self.db, self.domain)
        self.assertFalse(is_correct, "'alice' must not match inside 'malice'")


class TestSeeding(unittest.TestCase):
    """
    OpenEnv's reset() accepts a seed. Task selection must honour it so an
    episode can be replayed exactly.
    """

    def test_same_seed_selects_the_same_task(self):
        env_a, env_b = SQLAnalystEnv(), SQLAnalystEnv()
        env_a.reset(seed=42)
        env_b.reset(seed=42)
        self.assertEqual(env_a.state()["task_id"], env_b.state()["task_id"])
        env_a.close()
        env_b.close()

    def test_reset_accepts_the_openenv_signature(self):
        env = SQLAnalystEnv()
        env.reset(seed=7, episode_id="ep-1")
        self.assertEqual(env.state()["seed"], 7)
        self.assertEqual(env.state()["episode_id"], "ep-1")
        env.close()

    def test_different_seeds_can_select_different_tasks(self):
        seen = set()
        for seed in range(30):
            env = SQLAnalystEnv()
            env.reset(seed=seed)
            seen.add(env.state()["task_id"])
            env.close()
        self.assertGreater(len(seen), 1, "seeding must not collapse to one task")


class TestStepBudget(unittest.TestCase):
    """
    Regression: step_count was incremented and the loop shield checked before
    the action ran, so the agent's MAX_STEPS-th action was silently discarded
    and only MAX_STEPS - 1 were usable.
    """

    def test_agent_gets_the_full_step_budget(self):
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")
        for _ in range(MAX_STEPS - 1):
            _, _, done, _ = env.step(Action(sql_query="SELECT 1"))
            self.assertFalse(done)
        # The MAX_STEPS-th action must still be executed, then end the episode.
        _, reward, done, info = env.step(Action(sql_query="SELECT COUNT(*) FROM users"))
        self.assertTrue(done)
        self.assertEqual(info["step_count"], MAX_STEPS)
        self.assertEqual(reward.value, REWARD_INFINITE_LOOP)
        env.close()

    def test_submitting_on_the_final_step_is_graded_not_penalised(self):
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")
        for _ in range(MAX_STEPS - 1):
            env.step(Action(sql_query="SELECT 1"))
        _, reward, done, info = env.step(Action(submit_answer="15"))
        self.assertTrue(done)
        self.assertTrue(info["success"], "a correct final-step answer must be graded")
        self.assertNotEqual(reward.value, REWARD_INFINITE_LOOP)
        env.close()


class TestActionContract(unittest.TestCase):
    """The Action model must accept exactly one of sql_query / submit_answer."""

    def test_both_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            Action(sql_query="SELECT 1", submit_answer="15")

    def test_neither_field_is_rejected(self):
        with self.assertRaises(ValueError):
            Action()


class TestHttpServer(unittest.TestCase):
    """
    The HTTP surface is how OpenEnv clients actually consume the environment,
    so a valid SELECT must succeed through it end to end.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from server.app import app
        except ImportError as exc:
            raise unittest.SkipTest(f"fastapi not installed: {exc}")
        cls.client = TestClient(app)

    def test_health_check(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_valid_select_succeeds_over_http(self):
        self.client.post("/reset", json={})
        body = self.client.post("/step", json={"sql_query": "SELECT COUNT(*) FROM users"}).json()
        self.assertEqual(
            body["observation"]["error_message"], "",
            "a valid SELECT must not error over HTTP",
        )
        self.assertEqual(body["reward"]["value"], REWARD_SUCCESSFUL_QUERY)
        self.assertIn("15", body["observation"]["last_query_result"])

    def test_mutation_is_denied_over_http(self):
        self.client.post("/reset", json={})
        body = self.client.post("/step", json={"sql_query": "REPLACE INTO users VALUES (1,'x','x','x','x')"}).json()
        self.assertEqual(body["reward"]["value"], REWARD_DESTRUCTIVE_ACTION)
        self.assertTrue(body["done"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
