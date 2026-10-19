"""Print the cheat suite's scoreboard.  Run:  python -m adversaries"""

import sys

from .suite import run_suite


def main() -> int:
    verdicts = run_suite()
    task = None
    for v in verdicts:
        ep = v.episode
        if ep.task_id != task:
            task = ep.task_id
            print(f"\n{task}")
            print(f"  {'category':<10} {'agent':<24} {'ok':<4} {'steps':>5} "
                  f"{'reward':>7} {'score':>6}  result")
        print(f"  {v.category:<10} {ep.agent:<24} {'yes' if ep.success else 'no':<4} "
              f"{ep.steps:>5} {ep.total_reward:>7.2f} {ep.final_score:>6.2f}  "
              f"{'pass' if v.problem is None else 'FAIL: ' + v.problem}")
    failed = [v for v in verdicts if v.problem]
    print(f"\n{len(verdicts) - len(failed)}/{len(verdicts)} rules held.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
