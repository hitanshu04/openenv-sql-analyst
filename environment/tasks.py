# environment/tasks.py
# Task definitions for SQL Data Analyst environment
# 3 Tasks: Easy (single table COUNT), Medium (JOIN + aggregation), Hard (subquery/ordering)

from dataclasses import dataclass
from typing import Any, List, Optional
import random


@dataclass(frozen=True)
class Task:
    """
    Represents a data analysis task for the agent.

    Ground truth is deliberately NOT stored as a literal. It is derived at
    runtime by executing `ground_truth_sql` against the same database the
    agent queries (see `resolve_ground_truth`). A hardcoded answer can drift
    silently from the data; a derived one cannot.

    Attributes:
        task_id: Unique identifier for the task
        difficulty: easy, medium, or hard
        question: The business question to answer
        ground_truth_sql: Reference SQL that PRODUCES the correct answer
        answer_domain_sql: Optional SQL listing every value the answer could
            legitimately take. Used to reject hedged answers that name several
            candidates at once. Only meaningful for categorical answers.
        description: Additional context about the task
    """
    task_id: str
    difficulty: str
    question: str
    ground_truth_sql: str
    description: str
    answer_domain_sql: Optional[str] = None


# ============================================
# TASK DEFINITIONS
# ============================================

TASK_EASY = Task(
    task_id="easy_user_count",
    difficulty="easy",
    question=(
        "How many users are registered in the system? "
        "Provide the total count as a single number."
    ),
    ground_truth_sql="SELECT COUNT(*) FROM users",
    description="Single table COUNT query on users table"
)

TASK_MEDIUM = Task(
    task_id="medium_usa_revenue",
    difficulty="medium",
    question=(
        "What is the total revenue (sum of total_amount) from purchases made by users in the USA? "
        "Provide the total as a number (rounded to 2 decimal places if needed)."
    ),
    ground_truth_sql="""
        SELECT ROUND(SUM(p.total_amount), 2) as total_revenue
        FROM purchases p
        JOIN users u ON p.user_id = u.user_id
        WHERE u.country = 'USA'
    """,
    description="Two-table JOIN with SUM aggregation filtered by country"
)

TASK_HARD = Task(
    task_id="hard_top_spender",
    difficulty="hard",
    question=(
        "Who is the top spender (user with highest total purchase amount)? "
        "Provide the username of the user who spent the most money in total."
    ),
    ground_truth_sql="""
        SELECT u.username
        FROM users u
        JOIN purchases p ON u.user_id = p.user_id
        GROUP BY u.user_id, u.username
        ORDER BY SUM(p.total_amount) DESC
        LIMIT 1
    """,
    # Every username is a plausible answer, so naming several of them is a
    # guess-list rather than an answer.
    answer_domain_sql="SELECT username FROM users",
    description="Complex query with JOIN, GROUP BY, ORDER BY, and LIMIT"
)


# List of all tasks
TASKS: List[Task] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]


def resolve_ground_truth(task: Task, db_engine: Any) -> Any:
    """
    Compute a task's correct answer by running its reference SQL.

    This is the single source of truth for grading. Because the answer comes
    from the same database the agent sees, the two can never disagree.

    Args:
        task: The task whose answer should be computed
        db_engine: An initialized DatabaseEngine

    Returns:
        The scalar answer produced by task.ground_truth_sql
    """
    return db_engine.execute_privileged(task.ground_truth_sql)


def resolve_answer_domain(task: Task, db_engine: Any) -> Optional[List[Any]]:
    """
    Compute the set of legitimate candidate answers for a task, if it has one.

    Args:
        task: The task whose answer domain should be computed
        db_engine: An initialized DatabaseEngine

    Returns:
        A list of candidate answers, or None for tasks with no fixed domain
        (numeric answers, where hedging is not expressible).
    """
    if not task.answer_domain_sql:
        return None
    return db_engine.execute_privileged_column(task.answer_domain_sql)


def get_task_by_id(task_id: str) -> Task:
    """
    Get a task by its ID.

    Args:
        task_id: The unique task identifier

    Returns:
        Task: The matching task

    Raises:
        ValueError: If task_id not found
    """
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task not found: {task_id}")


def get_task_by_difficulty(difficulty: str) -> Task:
    """
    Get a task by difficulty level.

    Args:
        difficulty: easy, medium, or hard

    Returns:
        Task: A task matching the difficulty

    Raises:
        ValueError: If difficulty not found
    """
    for task in TASKS:
        if task.difficulty == difficulty:
            return task
    raise ValueError(f"No task found for difficulty: {difficulty}")


def get_random_task(seed: Optional[int] = None) -> Task:
    """
    Get a random task from the available tasks.

    Args:
        seed: Optional seed. Selection uses a local Random instance rather
            than the global module state, so a seeded reset is reproducible
            and does not perturb (or get perturbed by) other callers.

    Returns:
        Task: A randomly selected task
    """
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(TASKS)


def get_all_tasks() -> List[Task]:
    """
    Get all available tasks.

    Returns:
        List[Task]: All defined tasks
    """
    return TASKS.copy()
