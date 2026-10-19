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


# A number as people write it: optional minus, optional "$", digits with
# optional thousands separators, optional decimals, optional ordinal suffix
# ("46th"). The sign and the decimals belong to the number, so "-15" and
# "15.5" are not read as 15, and "15." at the end of a sentence is 15.
# "13,14,15" is three numbers, not 131,415.
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(-?)\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:st|nd|rd|th)?(?!\w)"
)


def extract_numbers(text: Any) -> List[float]:
    """Every number written in `text`, in order."""
    return [
        float(sign + whole.replace(",", "") + (decimals or ""))
        for sign, whole, decimals in NUMBER_PATTERN.findall(str(text))
    ]


def compare_numeric(submitted: Any, truth: float) -> Tuple[bool, float]:
    """
    Grade a numeric answer: it must contain exactly one distinct number, and
    that number is the answer. (Decision D8 in the v2 audit.)

    Whole-word matching, which categorical answers use, let a numeric answer
    win by containing the right digits anywhere: "0 1 2 ... 45" and
    "2204.87 or 2425.36" were graded correct, and so were "-15" and "15.5"
    in a sentence. Numeric tasks have no answer domain to catch a guess-list,
    so the guard is "one number only", which is what the questions ask for.
    """
    numbers = set(extract_numbers(submitted))
    if len(numbers) != 1:
        return False, MIN_SCORE  # no number, or a guess-list
    value = numbers.pop()
    if abs(value - truth) < 0.01:
        return True, MAX_SCORE
    # Partial credit for being close (within 10%)
    if truth != 0 and abs(value - truth) / abs(truth) < 0.1:
        return False, 0.5
    return False, MIN_SCORE


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

    Numeric ground truths are graded by compare_numeric: the answer is the one
    number the submission contains.

    Categorical answers are matched from strictest to loosest:
      1. exact match after normalization
      2. whole-word containment, so a natural-language answer such as
         "The top spender is alice." still counts

    Step 2 is guarded: if the submission names more than one value from
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
    if isinstance(ground_truth, (int, float)) and not isinstance(ground_truth, bool):
        return compare_numeric(submitted, float(ground_truth))

    # 1. Direct string comparison after normalization
    if normalize_value(submitted) == normalize_value(ground_truth):
        return True, MAX_SCORE  # 0.99 instead of 1.0

    # 2. Whole-word containment, rejected if the answer hedges across candidates
    if contains_whole_word(submitted, ground_truth):
        if answer_domain and count_domain_matches(submitted, answer_domain) > 1:
            return False, MIN_SCORE  # listing every option is not answering
        return True, MAX_SCORE  # 0.99 instead of 1.0

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

    A submission is always graded as a VALUE, never executed as SQL. The task
    asks for the value itself, and the agent has sql_query actions for running
    queries. The previous version executed any answer containing SELECT, FROM,
    WHERE, JOIN, GROUP or ORDER, so a correct sentence such as "The total
    revenue from USA customers is 2204.87" failed as broken SQL and scored
    0.01. (Decision D1 in the v2 audit.)

    Args:
        submitted_answer: The agent's submitted answer string
        ground_truth: The expected correct answer
        db_engine: Unused. Kept so existing callers keep working.
        answer_domain: Legitimate candidate answers, used to reject hedging

    Returns:
        Tuple[bool, float]: (is_correct, score)
            - is_correct: True if answer is correct
            - score: Value STRICTLY between 0 and 1
    """
    if not submitted_answer or not submitted_answer.strip():
        return False, MIN_SCORE  # 0.01 instead of 0.0

    return compare_values(submitted_answer.strip(), ground_truth, answer_domain)


def calculate_final_score(
    is_correct: bool,
    grading_score: float,
    total_steps: int,
    max_steps: int = 15,
    verified: bool = True,
) -> float:
    """
    Calculate the final score for a task.

    Scoring factors:
    - Correctness is primary
    - Efficiency bonus for fewer steps, only for a verified answer
    - Partial credit is preserved for near-miss numeric answers

    Args:
        is_correct: Whether the answer was correct
        grading_score: Score returned by the grader. For an incorrect answer
            this carries any partial credit (e.g. 0.5 for within 10%), which
            would otherwise be computed and then discarded.
        total_steps: Number of steps taken
        max_steps: Maximum allowed steps
        verified: Whether the agent read every table the task needs before
            answering. A correct answer without that evidence still counts as
            correct, but earns no efficiency bonus: otherwise a lucky
            one-step guess (0.96) outranked every agent that checked the data.
            (Decision D9 in the v2 audit.)

    Returns:
        float: Final score STRICTLY between 0 and 1
    """
    if not is_correct:
        # Honour the grader's partial credit instead of flattening every
        # wrong answer to the minimum score.
        return clamp_score(grading_score)

    # Base score for correct answer
    base_score = 0.7
    if not verified:
        return base_score

    # Efficiency bonus (up to 0.28 to stay under 0.99)
    # Fewer steps = higher bonus
    efficiency_ratio = 1.0 - (total_steps / max_steps)
    efficiency_bonus = max(0.0, efficiency_ratio * 0.28)

    final_score = base_score + efficiency_bonus

    # CRITICAL: Ensure score is STRICTLY between 0 and 1
    return clamp_score(final_score)
