#!/usr/bin/env python3
# inference.py
# Baseline Inference Script for OpenEnv SQL Analyst
# Uses OpenAI API client to run model against the environment
#
# HACKATHON REQUIREMENT: Must run ALL 3 tasks and output [START]/[STEP]/[END] for each

import os
import sys
import json
from typing import Optional, List, Tuple

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from environment.env import SQLAnalystEnv
from environment.models import Action
from environment.tasks import TASKS


# Environment configuration
BENCHMARK_NAME = "sql_analyst"
MAX_STEPS = 15


# ============================================
# SYSTEM PROMPT - Note: curly braces are escaped as {{ }}
# ============================================
SYSTEM_PROMPT = """You are an expert SQL Data Analyst AI agent. Your task is to answer business questions by querying a SQLite database.

You have two possible actions each turn:
1. Execute a SQL query to explore the data: {{"sql_query": "SELECT ..."}}
2. Submit your final answer: {{"submit_answer": "your answer"}}

IMPORTANT RULES:
- Only use SELECT queries. INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE are blocked.
- Explore the data step by step before submitting your final answer.
- Your final answer should be just the value requested (a number, name, etc.), not a SQL query.
- Respond with ONLY a valid JSON object, no other text.

DATABASE SCHEMA:
{schema_info}

CURRENT QUESTION:
{current_question}

LAST QUERY RESULT:
{last_query_result}

{error_section}

Respond with a JSON object containing either "sql_query" or "submit_answer"."""


def format_action_str(action: Action) -> str:
    """Format action for logging."""
    if action.sql_query:
        # Truncate long queries for logging
        query = action.sql_query.replace("\n", " ").strip()
        if len(query) > 50:
            query = query[:47] + "..."
        return f"sql_query={query}"
    elif action.submit_answer:
        answer = str(action.submit_answer).strip()
        if len(answer) > 30:
            answer = answer[:27] + "..."
        return f"submit_answer={answer}"
    return "invalid_action"


def extract_response_text(message) -> str:
    """
    Pull the usable text out of a chat completion message.

    Reasoning models may return an empty `content` and place their answer in a
    separate `reasoning` field, so a harness that reads only `content` sees
    nothing and records a parse error on every turn. Prefer content, fall back
    to reasoning.

    Args:
        message: The message object from a chat completion choice

    Returns:
        str: Best-effort response text, possibly empty
    """
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content

    try:
        return (message.model_dump().get("reasoning") or "").strip()
    except Exception:
        return ""


def iter_json_objects(text: str):
    """
    Yield every balanced {...} span in `text`, outermost first.

    Models wrap JSON in prose, in markdown fences, or in reasoning narration.
    Scanning for balanced braces handles all three without needing to guess at
    the wrapper format.
    """
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def parse_model_response(response_text: str) -> Optional[Action]:
    """
    Parse the model's response into an Action.

    Args:
        response_text: The raw text response from the model

    Returns:
        Action or None if parsing fails
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strip a markdown fence if the whole response is wrapped in one
    if "```" in text:
        fenced = text.split("```")
        for chunk in fenced:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                text = chunk
                break

    # Try the whole payload first, then any balanced JSON object inside it
    candidates = [text] + list(iter_json_objects(text))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(data, dict):
            continue
        if data.get("sql_query") is None and data.get("submit_answer") is None:
            continue

        try:
            return Action(
                sql_query=data.get("sql_query"),
                submit_answer=data.get("submit_answer"),
            )
        except ValueError:
            continue

    return None


def run_single_task(
    client: OpenAI, model_name: str, task_id: str
) -> Tuple[bool, float]:
    """
    Run inference for a single task.

    Args:
        client: OpenAI client configured with LiteLLM proxy
        model_name: Model identifier
        task_id: The specific task to run

    Returns:
        Tuple of (success, final_score)
    """
    # Initialize environment and reset with specific task
    env = SQLAnalystEnv()
    observation = env.reset(task_id=task_id)

    # Get task info from state
    state = env.state()
    task_name = state.get("task_id", "unknown")

    # Track rewards and steps
    rewards: List[float] = []
    step_num = 0
    done = False
    success = False
    final_score = 0.01  # Default to MIN_SCORE (strictly > 0)

    # ============================================
    # [START] LOG - EXACT FORMAT REQUIRED
    # ============================================
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}")

    # Conversation carried across turns so the agent can see what it already
    # tried. Without this the agent has no memory between steps.
    messages = [
        {
            "role": "system",
            "content": "You are a SQL expert. Respond only with valid JSON.",
        },
        {
            "role": "user",
            "content": SYSTEM_PROMPT.format(
                schema_info=observation.schema_info,
                current_question=observation.current_question,
                last_query_result=observation.last_query_result,
                error_section="",
            ),
        },
    ]

    try:
        while not done and step_num < MAX_STEPS:
            step_num += 1

            try:
                # Call the model with the full conversation so far.
                #
                # Rebuilding a single-turn prompt each step (the previous
                # behaviour) left the model with no memory of its own actions.
                # At temperature 0 that is a deterministic loop: it re-issues
                # the same query every turn until the step shield fires.
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=500,
                )

                # Extract response text (handles reasoning models that leave
                # `content` empty and put the payload in `reasoning`)
                response_text = extract_response_text(response.choices[0].message)

                # Record what the model said, so it can see its own history
                messages.append(
                    {"role": "assistant", "content": response_text or "(empty response)"}
                )

                # Parse into Action
                action = parse_model_response(response_text)

                if action is None:
                    # A malformed model response must not advance the
                    # environment. Substituting a placeholder query (the
                    # previous behaviour) paid +0.1 for a model failure AND
                    # fed its result back as context, which misled the model
                    # into answering with that placeholder. Instead: surface
                    # the parse error, score it zero, and let the model retry.
                    # The step budget bounds how often this can happen.
                    observation = observation.model_copy(
                        update={
                            "error_message": (
                                "Your previous response was not valid JSON. Reply with "
                                'exactly one of {"sql_query": "..."} or '
                                '{"submit_answer": "..."} and nothing else.'
                            )
                        }
                    )
                    messages.append(
                        {"role": "user", "content": observation.error_message}
                    )
                    rewards.append(0.0)
                    print(
                        f"[STEP]  step={step_num} action=parse_error reward=0.00 "
                        f"done=false error=parse_error"
                    )
                    continue

                error_msg = "null"

                # Execute action in environment
                observation, reward, done, info = env.step(action)

                # Feed the outcome back into the conversation
                outcome = observation.error_message or observation.last_query_result
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"RESULT OF YOUR LAST ACTION:\n{outcome}\n\n"
                            "If this answers the question, respond with "
                            '{"submit_answer": "..."}. Otherwise respond with '
                            'another {"sql_query": "..."}. JSON only.'
                        ),
                    }
                )

                # Track reward
                reward_value = reward.value
                rewards.append(reward_value)

                # Check for errors in observation
                if observation.error_message:
                    error_msg = observation.error_message.replace("\n", " ")[:50]

                # ============================================
                # [STEP] LOG - EXACT FORMAT REQUIRED
                # ============================================
                action_str = format_action_str(action)
                done_str = "true" if done else "false"
                print(
                    f"[STEP]  step={step_num} action={action_str} reward={reward_value:.2f} done={done_str} error={error_msg}"
                )

                # Update final results
                if done:
                    success = info.get("success", False)
                    final_score = info.get("final_score", 0.01)
                    # Ensure score is strictly between 0 and 1
                    if final_score <= 0.0:
                        final_score = 0.01
                    if final_score >= 1.0:
                        final_score = 0.99

            except Exception as e:
                # Handle API or other errors
                error_msg = str(e).replace("\n", " ")[:50]
                print(
                    f"[STEP]  step={step_num} action=error reward=0.00 done=false error={error_msg}"
                )
                rewards.append(0.0)

                # Try to continue with a simple action
                try:
                    action = Action(submit_answer="error")
                    observation, reward, done, info = env.step(action)
                    success = info.get("success", False)
                    final_score = info.get("final_score", 0.01)
                    # Ensure score is strictly between 0 and 1
                    if final_score <= 0.0:
                        final_score = 0.01
                    if final_score >= 1.0:
                        final_score = 0.99
                except Exception:
                    done = True
                    success = False
                    final_score = 0.01

    finally:
        # ============================================
        # [END] LOG - EXACT FORMAT REQUIRED
        # MUST ALWAYS BE EMITTED, EVEN ON EXCEPTION
        # ============================================
        success_str = "true" if success else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"
        print(
            f"[END]   success={success_str} steps={step_num} score={final_score:.2f} rewards={rewards_str}"
        )

        # Cleanup
        env.close()

    return success, final_score


def run_inference():
    """
    Run the baseline inference loop for ALL tasks.

    This function:
    1. Initializes the OpenAI client with injected LiteLLM proxy credentials
    2. Runs the model against EACH of the 3 tasks
    3. Outputs structured logs in the exact required format for each task

    HACKATHON REQUIREMENT: Must run all 3 tasks to pass "3+ tasks with graders" check
    """
    # ============================================
    # CONFIGURATION - Read env vars at runtime
    # Must use the injected API_BASE_URL and API_KEY from LiteLLM proxy
    # ============================================
    api_base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    # Initialize OpenAI client with injected credentials
    client = OpenAI(base_url=api_base_url, api_key=api_key)

    # ============================================
    # RUN ALL 3 TASKS
    # Hackathon requires: "3+ tasks with graders"
    # Validator: "Enumerate tasks, run each grader, verify scores"
    # ============================================
    results = []

    for task in TASKS:
        task_success, task_score = run_single_task(client, model_name, task.task_id)
        results.append((task.task_id, task_success, task_score))

    # Return overall success (all tasks passed) and average score
    all_success = all(r[1] for r in results)
    avg_score = sum(r[2] for r in results) / len(results) if results else 0.01

    return all_success, avg_score


def main():
    """Main entry point."""
    # Let exceptions propagate so validator can see real errors
    # But ensure [END] is always emitted (handled in run_single_task via finally)
    success, score = run_inference()
    sys.exit(0)


if __name__ == "__main__":
    main()
