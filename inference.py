import os
import re
import json
import textwrap
from typing import List
from openai import OpenAI
from client import SQLAnalystClient
from env import Action as SQLAction

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)
FALLBACK_ACTION = "noop()"
MAX_STEPS = 20

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a SQL Data Analyst Agent.
    Your goal is to answer business questions by writing and executing SQL queries.
    Reply with exactly one action string.
    The action must be a valid SQL command such as:
    - execute_sql('SELECT * FROM users')
    - submit_answer('42')
    - noop()
    Use single quotes around string arguments.
    Do not include explanations or additional text.
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = getattr(
        observation, "question", observation.get("question", "(not provided)")
    )
    schema = getattr(
        observation,
        "schema_summary",
        observation.get("schema_summary", "(none detected)"),
    )
    last_error = getattr(observation, "last_error", observation.get("last_error", None))
    error_note = "Yes" if last_error else "No"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Database Schema: {schema}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}
        Reply with exactly one SQL action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            action = re.sub(r"\s+", " ", action)
            return action

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def extract_sql_or_answer(action_str: str):
    """Extract sql_query or submit_answer from action string like execute_sql('SELECT...')"""
    action_str = action_str.strip()

    if action_str.startswith("execute_sql(") or action_str.startswith("submit_answer("):
        match = re.search(r"\((.*)\)", action_str)
        if match:
            content = match.group(1).strip()
            # Remove outer quotes if present
            if (content.startswith("'") and content.endswith("'")) or (
                content.startswith('"') and content.endswith('"')
            ):
                content = content[1:-1]

            if action_str.startswith("execute_sql("):
                return content, None
            else:
                return None, content

    if action_str == "noop()":
        return None, None

    # Default: treat as SQL query
    return action_str, None


def main():
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    env_url = os.environ.get("OPENENV_URL")

    if not api_key:
        print("Error: Set API_KEY, HF_TOKEN, or OPENAI_API_KEY environment variable")
        return

    client = OpenAI(base_url=base_url, api_key=api_key)

    tasks = ["monthly_signups", "top_revenue_category", "churn_analysis"]

    for task_id in tasks:
        print(
            f" {json.dumps({'task_id': task_id, 'task_name': task_id, 'difficulty': 'curriculum'})}"
        )

        history: List[str] = []

        # Use local environment instead of HTTP
        from env import SQLAnalystEnv as LocalEnv

        env = LocalEnv(task_id=task_id)
        result = env.reset()
        observation = result.observation
        total_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, observation, history)

            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = FALLBACK_ACTION

            action_str = parse_model_action(response_text)

            sql_query, submit_answer = extract_sql_or_answer(action_str)

            if submit_answer:
                action = SQLAction(submit_answer=submit_answer)
            elif sql_query:
                action = SQLAction(sql_query=sql_query)
            else:
                action = SQLAction(sql_query="SELECT 1")

            result = env.step(action)
            observation = result.observation
            reward = result.reward or 0.0
            total_reward += reward

            print(
                f" {json.dumps({'step': step, 'action': action_str, 'reward': reward, 'done': result.done})}"
            )

            error_flag = " ERROR" if observation.last_error else ""
            history_line = (
                f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            )
            history.append(history_line)

        print(
            f" {json.dumps({'total_steps': step, 'final_reward': total_reward, 'task_score': result.info.get('task_score', 0.0)})}"
        )

        avg_score = total_reward
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_id}")
        print(f"FINAL REWARD: {avg_score:.3f}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
