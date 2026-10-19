"""Play scripted agents against the environment and record what they earned."""

from dataclasses import dataclass
from typing import List, Optional

from environment.db_engine import DatabaseEngine
from environment.env import SQLAnalystEnv
from environment.models import Action
from environment.tasks import Task, resolve_answer_domain, resolve_ground_truth

from .agents import Agent, TaskFacts

# Everything a mutation could change: row counts, values, the schema, and
# attached databases. Run with privilege, so it can read pragma functions.
FINGERPRINT_SQL = """
    SELECT (SELECT COUNT(*) FROM users) || '|' ||
           (SELECT COUNT(*) FROM products) || '|' ||
           (SELECT COUNT(*) FROM purchases) || '|' ||
           (SELECT TOTAL(total_amount) FROM purchases) || '|' ||
           (SELECT GROUP_CONCAT(username) FROM users) || '|' ||
           (SELECT COUNT(*) FROM sqlite_master) || '|' ||
           (SELECT COUNT(*) FROM pragma_database_list)
"""


@dataclass(frozen=True)
class Episode:
    agent: str
    task_id: str
    steps: int
    first_reward: float
    total_reward: float
    final_score: float
    success: bool
    done: bool
    data_unchanged: bool


def facts_for(task: Task) -> TaskFacts:
    db = DatabaseEngine()
    db.initialize()
    try:
        return TaskFacts(
            task_id=task.task_id,
            reference_sql=task.ground_truth_sql.strip(),
            answer=resolve_ground_truth(task, db),
            domain=resolve_answer_domain(task, db),
            needed_tables=db.tables_read_by(task.ground_truth_sql),
        )
    finally:
        db.close()


def play(name: str, agent: Agent, task: Task) -> Optional[Episode]:
    """Run one agent on one task. Returns None if the agent doesn't apply."""
    actions: Optional[List[Action]] = agent(facts_for(task))
    if actions is None:
        return None

    env = SQLAnalystEnv()
    env.reset(task_id=task.task_id)
    before = env.db_engine.execute_privileged(FINGERPRINT_SQL)

    rewards: List[float] = []
    done, info = False, {}
    for action in actions:
        _, reward, done, info = env.step(action)
        rewards.append(reward.value)
        if done:
            break

    after = env.db_engine.execute_privileged(FINGERPRINT_SQL)
    env.close()
    return Episode(
        agent=name,
        task_id=task.task_id,
        steps=len(rewards),
        first_reward=rewards[0],
        total_reward=round(sum(rewards), 4),
        final_score=round(info.get("final_score", 0.0), 4),
        success=bool(info.get("success")),
        done=done,
        data_unchanged=(before == after),
    )
