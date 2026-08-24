#!/usr/bin/env python3
"""
demo.py - Run the environment end to end with a scripted agent.

No API key, no model, no network. This exists so that anyone evaluating the
repository can see the environment actually behave in about ten seconds,
rather than having to take the README's word for it.

It walks through the four properties that matter:

  1. Reward shaping        - what the agent earns for each kind of action
  2. Read-only enforcement - a mutation attempt is refused by SQLite itself
  3. Derived ground truth  - the answer comes from the task's reference SQL
  4. Seeded reproducibility - the same seed replays the same episode

Run:  python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import SQLAnalystEnv, MAX_STEPS
from environment.models import Action
from environment.tasks import TASKS, resolve_ground_truth


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def play(env, action: Action, label: str) -> None:
    """Execute one action and print what the environment returned."""
    obs, reward, done, info = env.step(action)
    print(f"  step {info['step_count']:<2} {label:<34} reward={reward.value:+.2f}  done={str(done).lower()}")

    if action.submit_answer:
        return  # the outcome is the score, printed by the caller

    detail = obs.error_message or obs.last_query_result.replace("\n", " ")
    if len(detail) > 46:
        detail = detail[:43] + "..."
    if detail:
        print(f"          -> {detail}")


def demo_episode() -> None:
    rule("1. A COMPLETE EPISODE  (task: hard_top_spender)")

    env = SQLAnalystEnv()
    obs = env.reset(seed=42, task_id="hard_top_spender")
    print(f"\n  Question: {obs.current_question}")
    print(f"  Ground truth, derived from the task's own reference SQL: "
          f"{env.state()['task_id']} -> {resolve_ground_truth(TASKS[2], env.db_engine)!r}\n")

    play(env, Action(sql_query="SELECT * FROM users LIMIT 3"), "explore the users table")
    play(env, Action(sql_query="SELECT * FRM purchases"), "typo: syntax error")
    play(env, Action(sql_query=(
        "SELECT u.username, SUM(p.total_amount) AS spend FROM users u "
        "JOIN purchases p ON u.user_id = p.user_id GROUP BY u.user_id "
        "ORDER BY spend DESC LIMIT 1"
    )), "the actual analysis")
    play(env, Action(submit_answer="alice"), "submit the answer")

    state = env.state()
    print(f"\n  success={state['success']}  final_score={state['final_score']:.2f}  "
          f"total_reward={state['total_reward']:+.2f}")
    env.close()


def demo_read_only() -> None:
    rule("2. READ-ONLY ENFORCEMENT  (SQLite authorizer, not a regex)")
    print()

    attacks = [
        ("DELETE FROM users", "classic mutation"),
        ("REPLACE INTO users (user_id,username,email,country,created_at) "
         "VALUES (1,'PWNED','x','X','2020')", "bypasses a keyword denylist"),
        ("CREATE TABLE evil AS SELECT * FROM users", "bypasses a keyword denylist"),
        ("ATTACH DATABASE ':memory:' AS side", "bypasses a keyword denylist"),
        ("PRAGMA query_only = OFF", "would defeat PRAGMA-based read-only"),
    ]

    for query, note in attacks:
        env = SQLAnalystEnv()
        env.reset(seed=1, task_id="easy_user_count")
        _, reward, done, _ = env.step(Action(sql_query=query))
        verdict = "REFUSED" if reward.value == -1.0 else "*** ALLOWED ***"
        print(f"  {verdict:<16} {query[:44]:<46} ({note})")
        env.close()

    # The false-positive direction matters just as much.
    env = SQLAnalystEnv()
    env.reset(seed=1, task_id="easy_user_count")
    _, reward, _, _ = env.step(
        Action(sql_query="SELECT COUNT(*) FROM users WHERE username = 'drop table'")
    )
    ok = "ALLOWED" if reward.value > 0 else "*** WRONGLY REFUSED ***"
    print(f"\n  {ok:<16} a legitimate SELECT with 'drop table' in a string literal")
    env.close()


def demo_grader() -> None:
    rule("3. THE GRADER CANNOT BE GAMED")
    print()

    cases = [
        ("alice", "the exact answer"),
        ("The top spender is alice.", "a natural-language answer"),
        ("alice bob charlie diana eve frank grace", "every candidate, hoping one sticks"),
    ]

    for answer, note in cases:
        env = SQLAnalystEnv()
        env.reset(seed=1, task_id="hard_top_spender")
        _, _, _, info = env.step(Action(submit_answer=answer))
        verdict = "correct" if info["success"] else "REJECTED"
        print(f"  {verdict:<10} score={info['final_score']:.2f}  {answer[:42]:<44} ({note})")
        env.close()


def demo_seeding() -> None:
    rule("4. SEEDED RESET REPLAYS THE SAME EPISODE")
    print()

    for seed in (42, 42, 7):
        env = SQLAnalystEnv()
        env.reset(seed=seed)
        print(f"  reset(seed={seed})".ljust(20) + f"-> task={env.state()['task_id']}")
        env.close()


def main() -> None:
    print("\nOpenEnv SQL Analyst - scripted demo (no API key required)")
    demo_episode()
    demo_read_only()
    demo_grader()
    demo_seeding()
    print(f"\n{'=' * 68}")
    print(f"Step budget per episode: {MAX_STEPS}.  Run the test suite with:")
    print("  python tests/test_environment.py")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
