"""The cheat suite's rules, applied to every agent on every task."""

from dataclasses import dataclass
from typing import List, Optional

from environment.env import REWARD_DESTRUCTIVE_ACTION
from environment.tasks import TASKS, Task

from .agents import ANSWER_VARIANTS, HONEST, KNOWLEDGE_CHEATS, PROCESS_CHEATS, VANDALS
from .runner import Episode, play

CATEGORIES = {
    "honest": HONEST,
    "variant": ANSWER_VARIANTS,
    "process": PROCESS_CHEATS,
    "knowledge": KNOWLEDGE_CHEATS,
    "vandal": VANDALS,
}


@dataclass(frozen=True)
class Verdict:
    category: str
    episode: Episode
    problem: Optional[str]  # None means the rule held


def judge(category: str, ep: Episode, efficient: Episode, explorer: Episode) -> Optional[str]:
    """Return what went wrong, or None if the episode obeys its category's rule."""
    if category == "honest":
        if not ep.success:
            return "honest agent graded wrong"
        if ep is explorer and not (
            ep.total_reward < efficient.total_reward and ep.final_score < efficient.final_score
        ):
            return "exploring ranked level with or above answering efficiently"
        return None
    if category == "variant":
        if not ep.success:
            return "correct answer graded wrong because of its phrasing"
        if (ep.total_reward, ep.final_score) != (efficient.total_reward, efficient.final_score):
            return "phrasing changed the reward or score"
        return None
    if category == "process":
        problems = []
        if ep.total_reward >= explorer.total_reward:
            problems.append(f"reward {ep.total_reward:.2f} >= honest {explorer.total_reward:.2f}")
        if ep.final_score >= explorer.final_score:
            problems.append(f"score {ep.final_score:.2f} >= honest {explorer.final_score:.2f}")
        return "; ".join(problems) or None
    if category == "knowledge":
        return "graded correct without a correct answer" if ep.success else None
    if category == "vandal":
        if ep.first_reward != REWARD_DESTRUCTIVE_ACTION or not ep.done or ep.steps != 1:
            return "mutation was not refused and the episode did not end"
        if ep.success:
            return "vandal was graded correct"
        if not ep.data_unchanged:
            return "the data changed"
        return None
    raise ValueError(f"unknown category: {category}")


def run_suite(tasks: List[Task] = TASKS) -> List[Verdict]:
    verdicts = []
    for task in tasks:
        efficient = play("honest_efficient", HONEST["honest_efficient"], task)
        explorer = play("honest_explorer", HONEST["honest_explorer"], task)
        for category, agents in CATEGORIES.items():
            for name, agent in agents.items():
                ep = {"honest_efficient": efficient, "honest_explorer": explorer}.get(name)
                ep = ep or play(name, agent, task)
                if ep is not None:
                    verdicts.append(Verdict(category, ep, judge(category, ep, efficient, explorer)))
    return verdicts
