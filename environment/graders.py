# environment/graders.py
# Deterministic grading system for SQL Data Analyst environment
# Implements type-agnostic normalization and SQL evaluation

from typing import Any, Tuple, Optional
import re


def normalize_value(value: Any) -> str:
    """
    Normalize a value for comparison.

    Type-Agnostic Normalization:
    - Strip whitespace
    - Lowercase strings
    - Handle numeric conversions

    Args:
        value: Any value to normalize

    Returns:
        str: Normalized string representation
    """
    if value is None:
        return ""

    # Convert to string first
    str_value = str(value).strip().lower()

    # Remove extra whitespace
    str_value = re.sub(r"\s+", " ", str_value)

    # Try to normalize numeric values
    try:
        # Try float first
        float_val = float(str_value)
        # Round to 2 decimal places for comparison
        return str(round(float_val, 2))
    except (ValueError, TypeError):
        pass

    return str_value


def extract_numeric(value: str) -> Optional[float]:
    """
    Extract a numeric value from a string.

    Args:
        value: String that may contain a number

    Returns:
        Optional[float]: Extracted number or None
    """
    # Remove common formatting
    cleaned = re.sub(r"[$,]", "", str(value).strip())

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def compare_values(submitted: Any, ground_truth: Any) -> Tuple[bool, float]:
    """
    Compare submitted answer to ground truth.

    Args:
        submitted: The agent's submitted answer
        ground_truth: The expected correct answer

    Returns:
        Tuple[bool, float]: (is_correct, score)
            - is_correct: True if answer matches
            - score: Value strictly between 0.0 and 1.0
    """
    # Normalize both values
    norm_submitted = normalize_value(submitted)
    norm_truth = normalize_value(ground_truth)

    # Direct string comparison after normalization
    if norm_submitted == norm_truth:
        return True, 0.99

    # Try numeric comparison for numeric ground truths
    if isinstance(ground_truth, (int, float)):
        submitted_num = extract_numeric(submitted)
        if submitted_num is not None:
            truth_num = float(ground_truth)
            # Allow small floating point tolerance
            if abs(submitted_num - truth_num) < 0.01:
                return True, 0.99
            # Partial credit for being close (within 10%)
            if truth_num != 0:
                error_pct = abs(submitted_num - truth_num) / abs(truth_num)
                if error_pct < 0.1:
                    return False, 0.05

    # Check if submitted answer contains the ground truth
    if norm_truth in norm_submitted:
        return True, 0.99

    return False, 0.01


def grade_sql_result(
    query_result: str, ground_truth: Any, is_error: bool
) -> Tuple[bool, float]:
    """
    Grade a SQL query result against ground truth.

    Args:
        query_result: The result string from executing the SQL query
        ground_truth: The expected correct answer
        is_error: Whether the query execution resulted in an error

    Returns:
        Tuple[bool, float]: (is_correct, score) - score strictly between 0.0 and 1.0
    """
    if is_error:
        return False, 0.01

    lines = query_result.strip().split("\n")

    data_lines = [l for l in lines if l.strip() and not l.startswith("|---")]

    if len(data_lines) < 2:
        return False, 0.01

    data_row = data_lines[1] if len(data_lines) > 1 else ""

    values = [v.strip() for v in data_row.split("|") if v.strip()]

    if not values:
        return False, 0.01

    for value in values:
        is_correct, score = compare_values(value, ground_truth)
        if is_correct:
            return True, score

    return False, 0.01


def grade_answer(
    submitted_answer: str, ground_truth: Any, db_engine: Any = None
) -> Tuple[bool, float]:
    """
    Grade the agent's submitted answer.

    Args:
        submitted_answer: The agent's submitted answer string
        ground_truth: The expected correct answer
        db_engine: Optional database engine for SQL evaluation

    Returns:
        Tuple[bool, float]: (is_correct, score) - score strictly between 0.0 and 1.0
    """
    if not submitted_answer or not submitted_answer.strip():
        return False, 0.01

    submitted = submitted_answer.strip()

    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER"]
    is_sql_query = any(keyword in submitted.upper() for keyword in sql_keywords)

    if is_sql_query and db_engine is not None:
        result, is_error = db_engine.execute_query(submitted)
        return grade_sql_result(result, ground_truth, is_error)

    return compare_values(submitted, ground_truth)


def calculate_final_score(
    is_correct: bool, total_steps: int, max_steps: int = 15
) -> float:
    """
    Calculate the final score for a task.

    Args:
        is_correct: Whether the answer was correct
        total_steps: Number of steps taken
        max_steps: Maximum allowed steps

    Returns:
        float: Final score strictly between 0.0 and 1.0
    """
    if not is_correct:
        return 0.01

    base_score = 0.7

    efficiency_ratio = 1.0 - (total_steps / max_steps)
    efficiency_bonus = max(0.0, efficiency_ratio * 0.3)

    final_score = base_score + efficiency_bonus

    # Ensure score is strictly between 0.0 and 1.0
    # Use 0.99 as max to stay strictly under 1.0
    return min(0.99, max(0.01, final_score))
