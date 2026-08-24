# environment/graders.py
# Deterministic grading system for SQL Data Analyst environment
# Implements type-agnostic normalization and SQL evaluation
# IMPORTANT: All scores must be STRICTLY between 0 and 1 (not 0.0, not 1.0)

from typing import Any, List, Optional, Tuple
import re


# Score boundaries - STRICTLY between 0 and 1
MIN_SCORE = 0.01  # Never return exactly 0.0
MAX_SCORE = 0.99  # Never return exactly 1.0


def clamp_score(score: float) -> float:
    """
    Clamp score to be strictly between 0 and 1.
    Hackathon requirement: scores must be in (0, 1) exclusive.
    """
    if score <= 0.0:
        return MIN_SCORE
    if score >= 1.0:
        return MAX_SCORE
    return score


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


def contains_whole_word(text: Any, word: Any) -> bool:
    """
    Check whether `word` appears in `text` as a standalone token.

    Used instead of plain substring containment so a ground truth of "alice"
    is not matched by "malice".
    """
    pattern = rf"(?<!\w){re.escape(str(word).strip().lower())}(?!\w)"
    return re.search(pattern, str(text).lower()) is not None


def count_domain_matches(text: Any, answer_domain: List[Any]) -> int:
    """
    Count how many distinct candidate answers appear in a submission.

    A final answer is singular by definition. An agent that names several
    candidates from the answer domain is hedging, not answering, and must not
    score full credit merely for having the right one somewhere in the list.

    Args:
        text: The agent's submitted answer
        answer_domain: Every value the answer could legitimately take

    Returns:
        int: Number of distinct domain values present as whole words
    """
    return sum(1 for candidate in set(answer_domain) if contains_whole_word(text, candidate))


def compare_values(
    submitted: Any, ground_truth: Any, answer_domain: Optional[List[Any]] = None
) -> Tuple[bool, float]:
    """
    Compare submitted answer to ground truth.

    Matching is deliberately ordered from strictest to loosest:
      1. exact match after normalization
      2. numeric comparison with tolerance and partial credit
      3. whole-word containment, so a natural-language answer such as
         "The top spender is alice." still counts

    Step 3 is guarded: if the submission names more than one value from
    `answer_domain`, it is a guess-list rather than an answer and is rejected.
    Without that guard an agent can submit every candidate and score full
    marks, which is reward hacking rather than analysis.

    Args:
        submitted: The agent's submitted answer
        ground_truth: The expected correct answer
        answer_domain: Optional set of legitimate candidate answers, used to
            detect hedging. Only meaningful for categorical answers.

    Returns:
        Tuple[bool, float]: (is_correct, score)
            - is_correct: True if answer matches
            - score: Value STRICTLY between 0 and 1
    """
    # Normalize both values
    norm_submitted = normalize_value(submitted)
    norm_truth = normalize_value(ground_truth)

    # 1. Direct string comparison after normalization
    if norm_submitted == norm_truth:
        return True, MAX_SCORE  # 0.99 instead of 1.0

    # 2. Try numeric comparison for numeric ground truths
    if isinstance(ground_truth, (int, float)):
        submitted_num = extract_numeric(submitted)
        if submitted_num is not None:
            truth_num = float(ground_truth)
            # Allow small floating point tolerance
            if abs(submitted_num - truth_num) < 0.01:
                return True, MAX_SCORE  # 0.99 instead of 1.0
            # Partial credit for being close (within 10%)
            if truth_num != 0:
                error_pct = abs(submitted_num - truth_num) / abs(truth_num)
                if error_pct < 0.1:
                    return False, 0.5

    # 3. Whole-word containment, rejected if the answer hedges across candidates
    if contains_whole_word(submitted, ground_truth):
        if answer_domain and count_domain_matches(submitted, answer_domain) > 1:
            return False, MIN_SCORE  # listing every option is not answering
        return True, MAX_SCORE  # 0.99 instead of 1.0

    return False, MIN_SCORE  # 0.01 instead of 0.0


def grade_sql_result(
    query_result: str, ground_truth: Any, is_error: bool
) -> Tuple[bool, float]:
    """
    Grade a SQL query result against ground truth.

    If the agent submits a SQL query as the final answer,
    this function evaluates the query result.

    Args:
        query_result: The result string from executing the SQL query
        ground_truth: The expected correct answer
        is_error: Whether the query execution resulted in an error

    Returns:
        Tuple[bool, float]: (is_correct, score)
    """
    if is_error:
        return False, MIN_SCORE  # 0.01 instead of 0.0

    # Parse the query result to extract values
    # Result format is markdown table: | col1 | col2 |
    lines = query_result.strip().split("\n")

    # Skip header and separator lines
    data_lines = [l for l in lines if l.strip() and not l.startswith("|---")]

    if len(data_lines) < 2:  # Need at least header + 1 data row
        return False, MIN_SCORE  # 0.01 instead of 0.0

    # Get the first data row (skip header)
    data_row = data_lines[1] if len(data_lines) > 1 else ""

    # Extract values from the row
    values = [v.strip() for v in data_row.split("|") if v.strip()]

    if not values:
        return False, MIN_SCORE  # 0.01 instead of 0.0

    # For single-value answers, compare the first value
    # For multi-column results, try each value
    for value in values:
        is_correct, score = compare_values(value, ground_truth)
        if is_correct:
            return True, score

    return False, MIN_SCORE  # 0.01 instead of 0.0


def grade_answer(
    submitted_answer: str,
    ground_truth: Any,
    db_engine: Any = None,
    answer_domain: Optional[List[Any]] = None,
) -> Tuple[bool, float]:
    """
    Grade the agent's submitted answer.

    This is the main grading function called by the environment.

    Args:
        submitted_answer: The agent's submitted answer string
        ground_truth: The expected correct answer
        db_engine: Optional database engine for SQL evaluation

    Returns:
        Tuple[bool, float]: (is_correct, score)
            - is_correct: True if answer is correct
            - score: Value STRICTLY between 0 and 1
    """
    if not submitted_answer or not submitted_answer.strip():
        return False, MIN_SCORE  # 0.01 instead of 0.0

    submitted = submitted_answer.strip()

    # Check if the submitted answer looks like a SQL query
    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER"]
    is_sql_query = any(keyword in submitted.upper() for keyword in sql_keywords)

    if is_sql_query and db_engine is not None:
        # Execute the SQL and grade the result
        result, status = db_engine.execute_query(submitted)
        return grade_sql_result(result, ground_truth, status != "ok")

    # Direct answer comparison
    return compare_values(submitted, ground_truth, answer_domain)


def calculate_final_score(
    is_correct: bool, grading_score: float, total_steps: int, max_steps: int = 15
) -> float:
    """
    Calculate the final score for a task.

    Scoring factors:
    - Correctness is primary
    - Efficiency bonus for fewer steps
    - Partial credit is preserved for near-miss numeric answers

    Args:
        is_correct: Whether the answer was correct
        grading_score: Score returned by the grader. For an incorrect answer
            this carries any partial credit (e.g. 0.5 for within 10%), which
            would otherwise be computed and then discarded.
        total_steps: Number of steps taken
        max_steps: Maximum allowed steps

    Returns:
        float: Final score STRICTLY between 0 and 1
    """
    if not is_correct:
        # Honour the grader's partial credit instead of flattening every
        # wrong answer to the minimum score.
        return clamp_score(grading_score)

    # Base score for correct answer
    base_score = 0.7

    # Efficiency bonus (up to 0.28 to stay under 0.99)
    # Fewer steps = higher bonus
    efficiency_ratio = 1.0 - (total_steps / max_steps)
    efficiency_bonus = max(0.0, efficiency_ratio * 0.28)

    final_score = base_score + efficiency_bonus

    # CRITICAL: Ensure score is STRICTLY between 0 and 1
    return clamp_score(final_score)
