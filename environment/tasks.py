# environment/tasks.py
# Task definitions for SQL Data Analyst environment. Each task tests one SQL
# skill; answers are always derived from data, never stored.

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

# Parameters shared with data/generate.py, which checks the data makes each
# question non-trivial (e.g. the named user's rank is not 1).
REPORT_YEAR = 2024
MIN_CATEGORIES = 4
RANK_TARGET = "karen"

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

TASK_UNSOLD = Task(
    task_id="medium_unsold_products",
    difficulty="medium",
    question=(
        "How many products have never been purchased? "
        "Provide the count as a single number."
    ),
    ground_truth_sql="""
        SELECT COUNT(*)
        FROM products p
        LEFT JOIN purchases pu ON pu.product_id = p.product_id
        WHERE pu.purchase_id IS NULL
    """,
    description="Anti-join: LEFT JOIN, keep rows with no match (IS NULL)"
)

TASK_MULTI_CATEGORY = Task(
    task_id="medium_multi_category_buyers",
    difficulty="medium",
    question=(
        f"How many users have bought products from at least {MIN_CATEGORIES} "
        "different categories? Provide the count as a single number."
    ),
    ground_truth_sql=f"""
        SELECT COUNT(*) FROM (
            SELECT pu.user_id
            FROM purchases pu
            JOIN products p ON p.product_id = pu.product_id
            GROUP BY pu.user_id
            HAVING COUNT(DISTINCT p.category) >= {MIN_CATEGORIES}
        )
    """,
    description="GROUP BY with HAVING over COUNT(DISTINCT ...), counted in a subquery"
)

TASK_BEST_MONTH = Task(
    task_id="medium_best_month",
    difficulty="medium",
    question=(
        f"Which month of {REPORT_YEAR} had the highest total revenue "
        f"(sum of total_amount)? Answer in YYYY-MM format, e.g. {REPORT_YEAR}-01."
    ),
    ground_truth_sql=f"""
        SELECT strftime('%Y-%m', purchase_date) AS month
        FROM purchases
        WHERE purchase_date LIKE '{REPORT_YEAR}-%'
        GROUP BY month
        ORDER BY SUM(total_amount) DESC
        LIMIT 1
    """,
    answer_domain_sql="SELECT DISTINCT strftime('%Y-%m', purchase_date) FROM purchases",
    description="Date bucketing with strftime, then aggregate and order"
)

TASK_SPEND_RANK = Task(
    task_id="hard_spend_rank",
    difficulty="hard",
    question=(
        "Rank the users who made at least one purchase by total spend "
        "(sum of total_amount), highest first, with rank 1 for the top spender. "
        f"What is {RANK_TARGET}'s rank? Provide the rank as a single number."
    ),
    ground_truth_sql=f"""
        SELECT spend_rank FROM (
            SELECT u.username,
                   RANK() OVER (ORDER BY SUM(pu.total_amount) DESC) AS spend_rank
            FROM users u
            JOIN purchases pu ON pu.user_id = u.user_id
            GROUP BY u.user_id, u.username
        )
        WHERE username = '{RANK_TARGET}'
    """,
    description="Window function RANK() over an aggregate, filtered in an outer query"
)

TASK_NET_REVENUE_COUNTRY = Task(
    task_id="hard_net_revenue_country",
    difficulty="hard",
    question=(
        "Which country's users generated the highest net revenue? Net revenue is "
        "the sum of total_amount minus the sum of refund_amount for refunded "
        "purchases. Provide the country name."
    ),
    ground_truth_sql="""
        SELECT u.country
        FROM users u
        JOIN purchases pu ON pu.user_id = u.user_id
        LEFT JOIN refunds r ON r.purchase_id = pu.purchase_id
        GROUP BY u.country
        ORDER BY SUM(pu.total_amount) - COALESCE(SUM(r.refund_amount), 0) DESC
        LIMIT 1
    """,
    answer_domain_sql="SELECT DISTINCT country FROM users",
    description="Three-table JOIN with a LEFT JOIN to refunds; gross and net rank differently"
)

TASK_TOP_BUYER_TIEBREAK = Task(
    task_id="hard_top_buyer_tiebreak",
    difficulty="hard",
    question=(
        "Which user made the most purchases (number of purchase rows)? If several "
        "users are tied, answer with the username that comes first alphabetically."
    ),
    ground_truth_sql="""
        SELECT u.username
        FROM users u
        JOIN purchases pu ON pu.user_id = u.user_id
        GROUP BY u.user_id, u.username
        ORDER BY COUNT(*) DESC, u.username ASC
        LIMIT 1
    """,
    answer_domain_sql="SELECT username FROM users",
    description="Aggregate with an explicit tie-break: the data has a tie at the top"
)


# --------------------------------------------------------------------------
# Harder tasks (P4b): real analyst problems that frontier models can miss.
# --------------------------------------------------------------------------

TASK_US_SIGNUPS = Task(
    task_id="hard_us_signups_dirty_country",
    difficulty="hard",
    question=(
        "How many signups (rows in the signups table) came from the United States? "
        "The country column is typed in by users. Provide the count as a single number."
    ),
    ground_truth_sql="""
        SELECT COUNT(*)
        FROM signups
        WHERE REPLACE(LOWER(TRIM(country)), '.', '') IN ('usa', 'us', 'united states')
    """,
    description="Dirty data: one country written many ways; explore the values before filtering"
)

TASK_DISTINCT_PEOPLE = Task(
    task_id="hard_distinct_real_signups",
    difficulty="hard",
    question=(
        "How many distinct people have signed up, excluding internal test accounts "
        "(emails at the sqlgym.test domain)? The same person may have signed up more "
        "than once. Provide the count as a single number."
    ),
    ground_truth_sql="""
        SELECT COUNT(DISTINCT LOWER(TRIM(email)))
        FROM signups
        WHERE LOWER(TRIM(email)) NOT LIKE '%@sqlgym.test'
    """,
    description="Deduplicate on a normalised key and apply a business exclusion rule"
)

TASK_CATEGORY_DROP = Task(
    task_id="hard_category_q2_q3_drop",
    difficulty="hard",
    question=(
        f"Which product category had the largest percentage drop in revenue (sum of "
        f"total_amount) from Q2 {REPORT_YEAR} (April-June) to Q3 {REPORT_YEAR} "
        "(July-September)? Provide the category name."
    ),
    ground_truth_sql=f"""
        SELECT category FROM (
            SELECT p.category,
                   SUM(CASE WHEN pu.purchase_date BETWEEN '{REPORT_YEAR}-04-01' AND '{REPORT_YEAR}-06-30'
                            THEN pu.total_amount ELSE 0 END) AS q2,
                   SUM(CASE WHEN pu.purchase_date BETWEEN '{REPORT_YEAR}-07-01' AND '{REPORT_YEAR}-09-30'
                            THEN pu.total_amount ELSE 0 END) AS q3
            FROM purchases pu
            JOIN products p ON p.product_id = pu.product_id
            GROUP BY p.category
        )
        ORDER BY (q3 - q2) / q2 ASC
        LIMIT 1
    """,
    answer_domain_sql="SELECT DISTINCT category FROM products",
    description="Conditional aggregation per quarter, then a ratio; absolute and relative drops disagree"
)

TASK_REPEAT_BUYERS = Task(
    task_id="hard_repeat_within_30_days",
    difficulty="hard",
    question=(
        "How many users made another purchase 1 to 30 days after their first ever "
        "purchase? Provide the count as a single number."
    ),
    ground_truth_sql="""
        SELECT COUNT(*) FROM (
            SELECT f.user_id
            FROM (SELECT user_id, MIN(purchase_date) AS first_date
                  FROM purchases GROUP BY user_id) f
            JOIN purchases pu ON pu.user_id = f.user_id
            WHERE julianday(pu.purchase_date) - julianday(f.first_date) BETWEEN 1 AND 30
            GROUP BY f.user_id
        )
    """,
    description="Anchor on each user's first purchase, then date arithmetic with julianday"
)

TASK_MEDIAN = Task(
    task_id="medium_median_order_value",
    difficulty="medium",
    question=(
        "What is the median total_amount across all purchases? Provide it as a "
        "number rounded to 2 decimal places."
    ),
    ground_truth_sql="""
        SELECT ROUND(AVG(total_amount), 2) FROM (
            SELECT total_amount FROM purchases
            ORDER BY total_amount
            LIMIT 2 - (SELECT COUNT(*) FROM purchases) % 2
            OFFSET ((SELECT COUNT(*) FROM purchases) - 1) / 2
        )
    """,
    description="Median without a MEDIAN function: ORDER BY with LIMIT/OFFSET; AVG is the trap"
)


# List of all tasks
TASKS: List[Task] = [
    TASK_EASY,
    TASK_MEDIUM,
    TASK_HARD,
    TASK_UNSOLD,
    TASK_MULTI_CATEGORY,
    TASK_BEST_MONTH,
    TASK_SPEND_RANK,
    TASK_NET_REVENUE_COUNTRY,
    TASK_TOP_BUYER_TIEBREAK,
    TASK_US_SIGNUPS,
    TASK_DISTINCT_PEOPLE,
    TASK_CATEGORY_DROP,
    TASK_REPEAT_BUYERS,
    TASK_MEDIAN,
]


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
