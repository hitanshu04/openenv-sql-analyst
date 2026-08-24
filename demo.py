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
      python demo.py --replay    # paced for screen recording
"""

import argparse
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import SQLAnalystEnv, MAX_STEPS
from environment.models import Action
from environment.tasks import TASKS, resolve_ground_truth


# --------------------------------------------------------------------------
# Presentation helpers
#
# Colour is emitted only when stdout is a terminal, so piping this into a file
# (as the README does) yields clean text, and CI logs stay readable. NO_COLOR
# is honoured per https://no-color.org.
# --------------------------------------------------------------------------
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

RESET = "\033[0m" if COLOR else ""
BOLD = "\033[1m" if COLOR else ""
DIM = "\033[2m" if COLOR else ""
GREEN = "\033[32m" if COLOR else ""
RED = "\033[31m" if COLOR else ""
CYAN = "\033[36m" if COLOR else ""
YELLOW = "\033[33m" if COLOR else ""

PACE = 0.0  # seconds between steps; raised by --replay


def pause(seconds: float = 1.0) -> None:
    if PACE:
        time.sleep(PACE * seconds)


def rule(title: str) -> None:
    print(f"\n{DIM}{'=' * 68}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{DIM}{'=' * 68}{RESET}")
    pause(0.6)


def reward_color(value: float) -> str:
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return DIM


def reward_bar(value: float) -> str:
    """A single block whose height tracks the magnitude of the reward."""
    blocks = {1.0: "█", 0.5: "▆", 0.1: "▃", 0.0: "▁"}
    magnitude = min(blocks, key=lambda k: abs(k - abs(value)))
    return blocks[magnitude]


def reward_trace(rewards: list) -> None:
    """Render the episode's reward signal as a compact trace."""
    if not rewards:
        return

    steps = "".join(f"{i + 1:>7}" for i in range(len(rewards)))
    values = "".join(
        f"{reward_color(r)}{r:>+7.2f}{RESET}" for r in rewards
    )
    bars = "".join(
        f"{reward_color(r)}{reward_bar(r):>7}{RESET}" for r in rewards
    )

    cumulative = []
    running = 0.0
    for r in rewards:
        running += r
        cumulative.append(running)
    cum = "".join(f"{c:>7.2f}" for c in cumulative)

    print(f"\n  {DIM}step  {RESET}{steps}")
    print(f"  {DIM}reward{RESET}{values}")
    print(f"        {bars}")
    print(f"  {DIM}cum.  {RESET}{cum}")


def play(env, action: Action, label: str, rewards: list) -> None:
    """Execute one action and print what the environment returned."""
    obs, reward, done, info = env.step(action)
    rewards.append(reward.value)

    colour = reward_color(reward.value)
    print(
        f"  {DIM}step {info['step_count']:<2}{RESET} {label:<34} "
        f"{colour}{BOLD}reward={reward.value:+.2f}{RESET}  "
        f"{DIM}done={str(done).lower()}{RESET}"
    )

    if not action.submit_answer:
        detail = obs.error_message or obs.last_query_result.replace("\n", " ")
        if len(detail) > 46:
            detail = detail[:43] + "..."
        if detail:
            colour = RED if obs.error_message else DIM
            print(f"          {colour}-> {detail}{RESET}")

    pause()


def demo_episode() -> None:
    rule("1. A COMPLETE EPISODE  (task: hard_top_spender)")

    env = SQLAnalystEnv()
    obs = env.reset(seed=42, task_id="hard_top_spender")
    truth = resolve_ground_truth(TASKS[2], env.db_engine)

    question = textwrap.fill(
        obs.current_question, width=88, initial_indent="", subsequent_indent="            "
    )
    print(f"\n  {CYAN}Question:{RESET} {question}")
    print(
        f"  {DIM}Ground truth, derived from the task's own reference SQL:{RESET} "
        f"{BOLD}{truth!r}{RESET}\n"
    )
    pause()

    rewards = []
    play(env, Action(sql_query="SELECT * FROM users LIMIT 3"),
         "explore the users table", rewards)
    play(env, Action(sql_query="SELECT * FRM purchases"),
         "typo: syntax error", rewards)
    play(env, Action(sql_query=(
        "SELECT u.username, SUM(p.total_amount) AS spend FROM users u "
        "JOIN purchases p ON u.user_id = p.user_id GROUP BY u.user_id "
        "ORDER BY spend DESC LIMIT 1"
    )), "the actual analysis", rewards)
    play(env, Action(submit_answer="alice"), "submit the answer", rewards)

    reward_trace(rewards)

    state = env.state()
    verdict = f"{GREEN}{BOLD}success{RESET}" if state["success"] else f"{RED}failed{RESET}"
    print(
        f"\n  {verdict}   final_score={BOLD}{state['final_score']:.2f}{RESET}   "
        f"{DIM}total_reward={state['total_reward']:+.2f}{RESET}"
    )
    env.close()
    pause(1.5)


def demo_read_only() -> None:
    rule("2. READ-ONLY ENFORCEMENT  (SQLite authorizer, not a regex)")
    print()

    attacks = [
        ("DELETE FROM users", "classic mutation"),
        ("REPLACE INTO users (user_id,username,email,country,created_at) "
         "VALUES (1,'PWNED','x','X','2020')", "regex denylist misses it"),
        ("CREATE TABLE evil AS SELECT * FROM users", "regex denylist misses it"),
        ("ATTACH DATABASE ':memory:' AS side", "regex denylist misses it"),
        ("PRAGMA query_only = OFF", "defeats PRAGMA read-only"),
    ]

    for query, note in attacks:
        env = SQLAnalystEnv()
        env.reset(seed=1, task_id="easy_user_count")
        _, reward, _, _ = env.step(Action(sql_query=query))
        blocked = reward.value == -1.0
        verdict = f"{GREEN}REFUSED{RESET}" if blocked else f"{RED}*** ALLOWED ***{RESET}"
        pad = 7 if blocked else 15
        print(f"  {verdict}{' ' * (17 - pad)}{query[:44]:<46} {DIM}({note}){RESET}")
        env.close()
        pause(0.5)

    # The false-positive direction matters just as much.
    env = SQLAnalystEnv()
    env.reset(seed=1, task_id="easy_user_count")
    _, reward, _, _ = env.step(
        Action(sql_query="SELECT COUNT(*) FROM users WHERE username = 'drop table'")
    )
    ok = reward.value > 0
    verdict = f"{GREEN}ALLOWED{RESET}" if ok else f"{RED}*** WRONGLY REFUSED ***{RESET}"
    print(f"\n  {verdict}{' ' * 10}a legitimate SELECT with 'drop table' in a string literal")
    env.close()
    pause(1.5)


def demo_grader() -> None:
    rule("3. THE GRADER CANNOT BE GAMED")
    print()

    cases = [
        ("alice", "the exact answer"),
        ("The top spender is alice.", "a natural-language answer"),
        ("alice bob charlie diana eve frank grace", "every candidate at once"),
    ]

    for answer, note in cases:
        env = SQLAnalystEnv()
        env.reset(seed=1, task_id="hard_top_spender")
        _, _, _, info = env.step(Action(submit_answer=answer))
        if info["success"]:
            verdict = f"{GREEN}correct {RESET}"
        else:
            verdict = f"{RED}REJECTED{RESET}"
        print(
            f"  {verdict}  score={info['final_score']:.2f}  "
            f"{answer[:42]:<44} {DIM}({note}){RESET}"
        )
        env.close()
        pause(0.7)

    pause(1.5)


def demo_seeding() -> None:
    rule("4. SEEDED RESET REPLAYS THE SAME EPISODE")
    print()

    for seed in (42, 42, 7):
        env = SQLAnalystEnv()
        env.reset(seed=seed)
        task = env.state()["task_id"]
        print(f"  {DIM}reset(seed={seed}){RESET}".ljust(30) + f"-> task={CYAN}{task}{RESET}")
        env.close()
        pause(0.6)


def main() -> None:
    global PACE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="pace the output for screen recording",
    )
    args = parser.parse_args()
    if args.replay:
        PACE = 0.55

    print(f"\n{BOLD}OpenEnv SQL Analyst{RESET} {DIM}- scripted demo (no API key required){RESET}")
    demo_episode()
    demo_read_only()
    demo_grader()
    demo_seeding()

    print(f"\n{DIM}{'=' * 68}{RESET}")
    print(f"Step budget per episode: {MAX_STEPS}.  Run the test suite with:")
    print(f"  {CYAN}python tests/test_environment.py{RESET}")
    print(f"{DIM}{'=' * 68}{RESET}\n")


if __name__ == "__main__":
    main()
