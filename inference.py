#!/usr/bin/env python3
# inference.py
# Baseline Inference Script for OpenEnv SQL Analyst
# Uses OpenAI API client to run model against the environment

import os
import sys
import json
from typing import Optional

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from environment.env import SQLAnalystEnv
from environment.models import Action


# Environment configuration
BENCHMARK_NAME = "sql_analyst"
MAX_STEPS = 15


# ============================================
# SYSTEM PROMPT
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


def parse_model_response(response_text: str) -> Optional[Action]:
    """
    Parse the model's response into an Action.

    Args:
        response_text: The raw text response from the model

    Returns:
        Action or None if parsing fails
    """
    try:
        # Clean the response
        text = response_text.strip()

        # Try to extract JSON from the response
        # Handle cases where model wraps JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Parse JSON
        data = json.loads(text)

        # Create Action
        return Action(
            sql_query=data.get("sql_query"), submit_answer=data.get("submit_answer")
        )
    except (json.JSONDecodeError, ValueError):
        return None


def run_inference():
    """
    Run the baseline inference loop.

    This function:
    1. Initializes the environment
    2. Runs the model against the environment
    3. Outputs structured logs in the exact required format
    """
    # ============================================
    # CONFIGURATION - Read env vars at runtime
    # Must use the injected API_BASE_URL and API_KEY
    # ============================================
    api_base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    # Initialize OpenAI client with injected credentials
    client = OpenAI(base_url=api_base_url, api_key=api_key)

    # Initialize environment
    env = SQLAnalystEnv()

    # Reset environment and get initial observation
    observation = env.reset()

    # Get task info from state
    state = env.state()
    task_name = state.get("task_id", "unknown")

    # ============================================
    # [START] LOG - EXACT FORMAT REQUIRED
    # ============================================
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}")

    # Track rewards and steps
    rewards = []
    step_num = 0
    done = False
    success = False
    final_score = 0.0

    while not done and step_num < MAX_STEPS:
        step_num += 1

        # Build the prompt
        error_section = ""
        if observation.error_message:
            error_section = f"ERROR FROM LAST ACTION:\n{observation.error_message}"

        prompt = SYSTEM_PROMPT.format(
            schema_info=observation.schema_info,
            current_question=observation.current_question,
            last_query_result=observation.last_query_result,
            error_section=error_section,
        )

        try:
            # Call the model via the injected LiteLLM proxy
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL expert. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Parse into Action
            action = parse_model_response(response_text)

            if action is None:
                # Failed to parse, try a simple query as fallback
                action = Action(sql_query="SELECT 1")
                error_msg = "parse_error"
            else:
                error_msg = "null"

            # Execute action in environment
            observation, reward, done, info = env.step(action)

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
                final_score = info.get("final_score", 0.0)

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
                final_score = info.get("final_score", 0.0)
            except:
                done = True
                success = False
                final_score = 0.0

    # ============================================
    # [END] LOG - EXACT FORMAT REQUIRED
    # ============================================
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END]   success={success_str} steps={step_num} score={final_score:.2f} rewards={rewards_str}"
    )

    # Cleanup
    env.close()

    return success, final_score


def main():
    """Main entry point."""
    # DO NOT catch exceptions here - let them propagate
    # so the validator can see real errors
    success, score = run_inference()
    sys.exit(0)


if __name__ == "__main__":
    main()
