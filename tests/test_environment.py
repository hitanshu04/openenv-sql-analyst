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

import json
import os
import re
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import db_engine as db_engine_module
from environment.db_engine import DatabaseEngine, QueryStatus, MAX_FETCH_ROWS
from environment.env import (
    MAX_STEPS,
    SQLAnalystEnv,
    REWARD_PROGRESS,
    REWARD_NO_PROGRESS,
    REWARD_SYNTAX_ERROR,
    REWARD_DESTRUCTIVE_ACTION,
    REWARD_INFINITE_LOOP,
)
from environment.models import Action
from environment.graders import calculate_final_score, grade_answer
from environment.tasks import TASKS, resolve_ground_truth, get_task_by_id
from data import generate


def truth(task_id):
    """A task's answer, derived from data, so tests never hardcode one."""
    db = DatabaseEngine()
    db.initialize()
    try:
        return resolve_ground_truth(get_task_by_id(task_id), db)
    finally:
        db.close()


class TestBenchmarkData(unittest.TestCase):
    """
    The v2 dataset is generated from a seed by data/generate.py. Every task's
    answer is computed twice, by its reference SQL and by an independent
    Python oracle over the generator's rows, and the two must agree.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate.build()

    def test_the_committed_data_is_exactly_what_the_generator_writes(self):
        with open(generate.OUT) as fh:
            self.assertEqual(fh.read(), generate.to_sql(self.dataset))

    def test_every_task_has_an_independent_oracle_that_agrees(self):
        self.assertEqual(set(generate.ORACLES), {task.task_id for task in TASKS})
        db = DatabaseEngine()
        db.initialize()
        try:
            for task in TASKS:
                with self.subTest(task=task.task_id):
                    from_sql = resolve_ground_truth(task, db)
                    from_python = generate.ORACLES[task.task_id](self.dataset)
                    if isinstance(from_python, float):
                        self.assertAlmostEqual(from_sql, from_python, places=2)
                    else:
                        self.assertEqual(from_sql, from_python)
        finally:
            db.close()

    def test_every_task_is_unambiguous_and_has_its_trap(self):
        self.assertEqual(generate.check(self.dataset), [])

    def test_each_trap_catches_its_naive_query(self):
        """The obvious-but-wrong query must give a different answer."""
        naive = {
            # Ignores refunds: ranks countries by gross revenue.
            "hard_net_revenue_country": """
                SELECT u.country FROM users u
                JOIN purchases pu ON pu.user_id = u.user_id
                GROUP BY u.country ORDER BY SUM(pu.total_amount) DESC LIMIT 1""",
            # Counts purchase rows instead of summing revenue.
            "medium_best_month": """
                SELECT strftime('%Y-%m', purchase_date) AS month FROM purchases
                GROUP BY month ORDER BY COUNT(*) DESC LIMIT 1""",
            # Trusts the free-text country column as written.
            "hard_us_signups_dirty_country": "SELECT COUNT(*) FROM signups WHERE country = 'USA'",
            # Dedupes on the raw email, so case and spacing variants count twice.
            "hard_distinct_real_signups": """
                SELECT COUNT(DISTINCT email) FROM signups
                WHERE email NOT LIKE '%@sqlgym.test'""",
            # Ranks by the absolute drop instead of the percentage drop.
            "hard_category_q2_q3_drop": """
                SELECT category FROM (
                    SELECT p.category,
                           SUM(CASE WHEN pu.purchase_date BETWEEN '2024-04-01' AND '2024-06-30'
                                    THEN pu.total_amount ELSE 0 END) AS q2,
                           SUM(CASE WHEN pu.purchase_date BETWEEN '2024-07-01' AND '2024-09-30'
                                    THEN pu.total_amount ELSE 0 END) AS q3
                    FROM purchases pu JOIN products p ON p.product_id = pu.product_id
                    GROUP BY p.category)
                ORDER BY q3 - q2 ASC LIMIT 1""",
            # Any two purchases 1-30 days apart, not measured from the first one.
            "hard_repeat_within_30_days": """
                SELECT COUNT(DISTINCT a.user_id) FROM purchases a
                JOIN purchases b ON a.user_id = b.user_id
                WHERE julianday(b.purchase_date) - julianday(a.purchase_date) BETWEEN 1 AND 30""",
            # The mean, not the median.
            "medium_median_order_value": "SELECT ROUND(AVG(total_amount), 2) FROM purchases",
            # Ignores the tie-break rule.
            "hard_top_buyer_tiebreak": """
                SELECT u.username FROM users u
                JOIN purchases pu ON pu.user_id = u.user_id
                GROUP BY u.user_id ORDER BY COUNT(*) DESC LIMIT 1""",
        }
        db = DatabaseEngine()
        db.initialize()
        try:
            for task_id, sql in naive.items():
                with self.subTest(task=task_id):
                    self.assertNotEqual(db.execute_privileged(sql), truth(task_id))
        finally:
            db.close()


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

    def test_a_correct_answer_is_graded_correct(self):
        """The end-to-end property the 2423.87 bug violated, on every task."""
        for task in TASKS:
            with self.subTest(task=task.task_id):
                env = SQLAnalystEnv()
                env.reset(task_id=task.task_id)
                _, _, done, info = env.step(Action(submit_answer=str(truth(task.task_id))))
                env.close()
                self.assertTrue(done)
                self.assertTrue(info["success"], "a correct answer must grade as correct")

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
        self.assertIn(str(truth("easy_user_count")), result)

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
        self.assertEqual(ok.value, REWARD_PROGRESS)
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
        _, reward, done, info = env.step(Action(submit_answer=str(truth("easy_user_count"))))
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
        # Pinned: the reward depends on which tables the task needs.
        self.client.post("/reset", json={"task_id": "easy_user_count"})
        body = self.client.post("/step", json={"sql_query": "SELECT COUNT(*) FROM users"}).json()
        self.assertEqual(
            body["observation"]["error_message"], "",
            "a valid SELECT must not error over HTTP",
        )
        self.assertEqual(body["reward"]["value"], REWARD_PROGRESS)
        self.assertIn(str(truth("easy_user_count")), body["observation"]["last_query_result"])

    def test_mutation_is_denied_over_http(self):
        self.client.post("/reset", json={})
        body = self.client.post("/step", json={"sql_query": "REPLACE INTO users VALUES (1,'x','x','x','x')"}).json()
        self.assertEqual(body["reward"]["value"], REWARD_DESTRUCTIVE_ACTION)
        self.assertTrue(body["done"])


class TestV2Regressions(unittest.TestCase):
    """
    Defects found in the v2 audit (16-17 Oct 2026) by running the environment.
    None of the 33 tests above covered them.

      1. The grader treats any answer containing SELECT/FROM/WHERE/JOIN/GROUP/
         ORDER as SQL and executes it, so a correct natural-language answer
         ("The total revenue from USA customers is 2204.87") scores 0.01.
      2. Every successful query earns +0.1, so stalling with SELECT 1 earns
         more total reward than answering efficiently (2.3 vs 1.1).
      3. openenv.yaml still declares the medium task's answer as 2423.87, the
         value the August audit proved wrong. The manifest drifted from data.
    """

    def test_correct_sentences_containing_sql_words_are_graded_correct(self):
        cases = {
            "easy_user_count": "Counting from the users table, there are {} users.",
            "medium_usa_revenue": "The total revenue from USA customers is {}",
            "hard_top_spender": "{} was selected as the top spender",
        }
        cases = {task_id: text.format(truth(task_id)) for task_id, text in cases.items()}
        for task_id, answer in cases.items():
            with self.subTest(task=task_id):
                env = SQLAnalystEnv()
                env.reset(task_id=task_id)
                _, _, done, info = env.step(Action(submit_answer=answer))
                env.close()
                self.assertTrue(done)
                self.assertTrue(
                    info["success"],
                    f"a correct answer was graded wrong: {answer!r}",
                )

    def _total_reward(self, queries, answer, task_id="easy_user_count"):
        env = SQLAnalystEnv()
        env.reset(task_id=task_id)
        total = 0.0
        for query in queries:
            _, reward, _, _ = env.step(Action(sql_query=query))
            total += reward.value
        _, reward, _, info = env.step(Action(submit_answer=answer))
        total += reward.value
        env.close()
        self.assertTrue(info["success"])
        return total

    def test_stalling_never_out_earns_answering_efficiently(self):
        answer = str(truth("easy_user_count"))
        stalling = self._total_reward(["SELECT 1"] * 13, answer)
        efficient = self._total_reward(["SELECT COUNT(*) FROM users"], answer)
        self.assertLess(
            stalling, efficient,
            f"stalling earned {stalling:.2f} vs {efficient:.2f} for the efficient run",
        )

    def test_rewording_a_query_cannot_farm_reward(self):
        """
        Found while fixing the stall bug: a rule that pays for a "new result"
        is farmable, because renaming a column or adding a constant makes every
        result look new. 13 such queries earned 2.3 vs 1.1 for the efficient run.
        """
        answer = str(truth("easy_user_count"))
        efficient = self._total_reward(["SELECT COUNT(*) FROM users"], answer)
        farms = {
            "aliases": [f"SELECT COUNT(*) AS c{k} FROM users" for k in range(13)],
            "arithmetic": [f"SELECT COUNT(*) + {k} FROM users" for k in range(13)],
        }
        for name, queries in farms.items():
            with self.subTest(farm=name):
                farmed = self._total_reward(queries, answer)
                self.assertLess(
                    farmed, efficient,
                    f"{name} farming earned {farmed:.2f} vs {efficient:.2f}",
                )

    def test_each_needed_table_is_paid_once(self):
        """Decision D2: +0.1 the first time a query reads each table the task's
        reference SQL needs; any other successful query costs 0.01."""
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")  # reference SQL reads: users
        steps = [
            ("SELECT 1", REWARD_NO_PROGRESS),                         # reads no table
            ("SELECT name FROM sqlite_master", REWARD_NO_PROGRESS),   # system table
            ("SELECT COUNT(*) FROM purchases", REWARD_NO_PROGRESS),   # not needed here
            ("SELECT COUNT(*) FROM users", REWARD_PROGRESS),          # first read of users
            ("SELECT COUNT(*) FROM users", REWARD_NO_PROGRESS),       # already paid
            ("SELECT COUNT(*) AS n FROM users", REWARD_NO_PROGRESS),  # reworded, already paid
        ]
        for query, expected in steps:
            with self.subTest(query=query):
                _, reward, _, _ = env.step(Action(sql_query=query))
                self.assertEqual(reward.value, expected)
        env.close()

    def test_one_join_is_paid_for_every_needed_table_it_reads(self):
        env = SQLAnalystEnv()
        env.reset(task_id="medium_usa_revenue")  # reference SQL reads: users, purchases
        _, reward, _, _ = env.step(Action(sql_query=(
            "SELECT SUM(p.total_amount) FROM purchases p "
            "JOIN users u ON p.user_id = u.user_id WHERE u.country = 'USA'"
        )))
        self.assertAlmostEqual(reward.value, 2 * REWARD_PROGRESS)
        _, reward, _, _ = env.step(Action(sql_query="SELECT * FROM users LIMIT 1"))
        self.assertEqual(reward.value, REWARD_NO_PROGRESS)
        env.close()

    def test_a_query_submitted_as_the_answer_is_not_executed(self):
        """Decision D1: answers are graded as values, so SQL is not an answer."""
        env = SQLAnalystEnv()
        env.reset(task_id="easy_user_count")
        _, _, done, info = env.step(Action(submit_answer="SELECT COUNT(*) FROM users"))
        env.close()
        self.assertTrue(done)
        self.assertFalse(info["success"], "a query is not the value the task asked for")

    def test_tables_read_are_reported_when_a_query_repeats(self):
        """
        Found by running the Docker image (Python 3.10): Python before 3.11
        reuses a prepared statement even with cached_statements=0, so the
        authorizer never saw a repeated query and reported no tables. The easy
        and medium tasks then had no required tables, and every answer counted
        as verified.
        """
        db = DatabaseEngine()
        db.initialize()
        sql = "SELECT COUNT(*) FROM users"
        try:
            self.assertEqual(db.tables_read_by(sql), {"users"})
            self.assertEqual(db.tables_read_by(sql), {"users"})
            db.execute_query(sql)
            self.assertEqual(db.last_query_tables, {"users"})
            db.execute_query(sql)
            self.assertEqual(db.last_query_tables, {"users"})
        finally:
            db.close()

    def test_a_numeric_answer_is_the_one_number_written(self):
        """Decision D8: parse numbers properly; exactly one distinct number allowed."""
        cases = [
            # (answer, ground truth, correct, score)
            ("15", 15, True, 0.99),
            ("There are 15 users.", 15, True, 0.99),
            ("15 users, so 15.", 15, True, 0.99),           # repeated, still one number
            ("$2,204.87", 2204.87, True, 0.99),
            ("2204.870", 2204.87, True, 0.99),
            ("The answer is -15.", 15, False, 0.01),        # a different number
            ("The answer is 15.5", 15, False, 0.5),         # same as bare "15.5"
            ("15 or 16", 15, False, 0.01),                   # hedging
            ("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", 15, False, 0.01),
            ("13,14,15", 15, False, 0.01),                   # a list, not 13,145
            ("fifteen", 15, False, 0.01),                    # no number at all
            ("46th", 46, True, 0.99),                        # a rank, written as one
            ("karen is ranked 3rd.", 3, True, 0.99),
            ("15users", 15, False, 0.01),                    # not a number token
        ]
        for answer, truth, correct, score in cases:
            with self.subTest(answer=answer):
                self.assertEqual(grade_answer(answer, truth), (correct, score))

    def test_a_correct_answer_without_checking_the_data_earns_no_bonus(self):
        """Decision D9 (E2): success stays truthful, but the efficiency bonus
        needs evidence: every table the task needs must have been read."""
        def finish(task_id, queries, answer):
            env = SQLAnalystEnv()
            env.reset(task_id=task_id)
            for query in queries:
                env.step(Action(sql_query=query))
            _, _, _, info = env.step(Action(submit_answer=answer))
            env.close()
            return info

        users = str(truth("easy_user_count"))
        blind = finish("easy_user_count", [], users)
        self.assertTrue(blind["success"])
        self.assertFalse(blind["verified"])
        self.assertAlmostEqual(blind["final_score"], 0.70)

        half_checked = finish(
            "medium_usa_revenue", ["SELECT * FROM users LIMIT 1"], str(truth("medium_usa_revenue"))
        )
        self.assertFalse(half_checked["verified"], "purchases was never read")
        self.assertAlmostEqual(half_checked["final_score"], 0.70)

        checked = finish("easy_user_count", ["SELECT COUNT(*) FROM users"], users)
        self.assertTrue(checked["verified"])
        self.assertGreater(checked["final_score"], blind["final_score"])

    def test_manifest_ground_truths_match_the_data(self):
        manifest = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openenv.yaml"
        )
        with open(manifest) as fh:
            text = fh.read()
        # A deliberately tiny parser: read only the task_id / ground_truth
        # pairs, so this test needs no YAML dependency.
        declared = dict(re.findall(
            r"-\s*task_id:\s*(\S+).*?ground_truth:\s*\"?([^\"\n]+)\"?", text, re.S
        ))
        self.assertEqual(set(declared), {task.task_id for task in TASKS})
        questions = dict(re.findall(
            r"-\s*task_id:\s*(\S+).*?question:\s*\"((?:[^\"\\]|\\.)*)\"", text, re.S
        ))

        db = DatabaseEngine()
        db.initialize()
        try:
            for task in TASKS:
                with self.subTest(task=task.task_id):
                    self.assertEqual(json.loads(f'"{questions[task.task_id]}"'), task.question)
                    derived = resolve_ground_truth(task, db)
                    stated = declared[task.task_id].strip()
                    if isinstance(derived, (int, float)):
                        self.assertAlmostEqual(float(stated), float(derived), places=2)
                    else:
                        self.assertEqual(stated, str(derived))
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
