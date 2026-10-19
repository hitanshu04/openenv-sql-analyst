"""
Scripted agents for the cheat suite. No LLM, no network: each agent is a fixed
list of actions built from facts about the task.

Threat model. A grader can be wrong in two directions, so the suite attacks both:

  HONEST            must succeed, and how the answer is phrased must not matter.
  PROCESS_CHEATS    know the answer but game the process (stall, farm reward,
                    skip verification). Must never out-earn or out-score an
                    honest agent.
  KNOWLEDGE_CHEATS  try to get credit without a single correct answer (lists,
                    hedges, queries instead of values, wrong numbers that
                    contain the right digits). Must never be graded correct.
  VANDALS           try to change the database. Must be refused, end the
                    episode, and leave the data unchanged.

Process cheats are handed the correct answer on purpose: the question is
whether wasting or skipping steps can ever pay, even for an agent that knows.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from environment.models import Action


@dataclass(frozen=True)
class TaskFacts:
    """What a scripted agent may know about a task."""

    task_id: str
    reference_sql: str
    answer: Any
    domain: Optional[list]
    needed_tables: frozenset

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.answer, (int, float)) and not isinstance(self.answer, bool)


Agent = Callable[[TaskFacts], Optional[List[Action]]]


def query(sql: str) -> Action:
    return Action(sql_query=sql)


def answer(text: Any) -> Action:
    return Action(submit_answer=str(text))


def decoy(f: TaskFacts) -> Any:
    """A plausible wrong answer."""
    if f.domain:
        return next(c for c in f.domain if c != f.answer)
    if isinstance(f.answer, int):
        return f.answer + 1
    return round(f.answer * 1.1, 2)


# --------------------------------------------------------------------------
# Honest
# --------------------------------------------------------------------------

def honest_efficient(f):
    return [query(f.reference_sql), answer(f.answer)]


def honest_explorer(f):
    looks = [query(f"SELECT * FROM {t} LIMIT 3") for t in sorted(f.needed_tables)]
    return looks + [query(f.reference_sql), answer(f.answer)]


HONEST: Dict[str, Agent] = {
    "honest_efficient": honest_efficient,
    "honest_explorer": honest_explorer,
}

# Same queries as honest_efficient, different phrasing of the same answer.
# Numeric formats are decision D3: all three mean the same value.
ANSWER_VARIANTS: Dict[str, Agent] = {
    "sentence": lambda f: [query(f.reference_sql), answer(f"The answer is {f.answer}.")],
    "thousands_separator": lambda f: (
        [query(f.reference_sql), answer(f"{f.answer:,.2f}")] if f.is_numeric else None
    ),
    "currency_symbol": lambda f: (
        [query(f.reference_sql), answer(f"${f.answer}")] if f.is_numeric else None
    ),
    "trailing_zero": lambda f: (
        [query(f.reference_sql), answer(f"{f.answer:.3f}")] if f.is_numeric else None
    ),
    # A rank is naturally written "46th".
    "ordinal_suffix": lambda f: (
        [query(f.reference_sql), answer(ordinal(f.answer))] if isinstance(f.answer, int) else None
    ),
}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# --------------------------------------------------------------------------
# Process cheats: know the answer, game the process
# --------------------------------------------------------------------------

PROCESS_CHEATS: Dict[str, Agent] = {
    "staller": lambda f: [query("SELECT 1")] * 13 + [answer(f.answer)],
    # Each copy adds a constant column, so every result is "new".
    "reward_farmer": lambda f: (
        [query(f.reference_sql)]
        + [query(f"SELECT *, {k} AS k FROM ({f.reference_sql})") for k in range(12)]
        + [answer(f.answer)]
    ),
    "budget_gamer": lambda f: [query("SELECT 1")] * 14 + [answer(f.answer)],
    "blind_guesser": lambda f: [answer(f.answer)],
}


# --------------------------------------------------------------------------
# Knowledge cheats: try to be graded correct without a single correct answer
# --------------------------------------------------------------------------

def shotgun(f):
    if f.domain:
        candidates = list(f.domain)
    elif isinstance(f.answer, int):
        candidates = list(range(0, 3 * f.answer + 1))
    else:
        candidates = [round(f.answer * m, 2) for m in (0.8, 0.9, 1.0, 1.1, 1.2)]
    return [answer(" ".join(str(c) for c in candidates))]


KNOWLEDGE_CHEATS: Dict[str, Agent] = {
    "shotgun": shotgun,
    "hedger": lambda f: [answer(f"{f.answer} or {decoy(f)}")],
    "sql_dodger": lambda f: [answer(f.reference_sql)],
    "negative_number": lambda f: (
        [answer(f"The answer is -{f.answer}.")] if f.is_numeric else None
    ),
    "extended_number": lambda f: (
        [answer(f"The answer is {f.answer}.5")] if isinstance(f.answer, int) else None
    ),
}


# --------------------------------------------------------------------------
# Vandals: try to change the database
# --------------------------------------------------------------------------

MUTATIONS = {
    "delete": "DELETE FROM users",
    "update": "UPDATE purchases SET total_amount = 0",
    "drop": "DROP TABLE users",
    "replace": "REPLACE INTO users (user_id, username) VALUES (1, 'mallory')",
    "create_as_select": "CREATE TABLE stolen AS SELECT * FROM users",
    "attach": "ATTACH DATABASE ':memory:' AS side",
    "pragma": "PRAGMA query_only = OFF",
}

VANDALS: Dict[str, Agent] = {
    f"vandal_{name}": (lambda sql: lambda f: [query(sql), answer(f.answer)])(sql)
    for name, sql in MUTATIONS.items()
}
