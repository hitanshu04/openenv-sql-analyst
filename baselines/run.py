"""
Run the baseline lineup on every task, one JSON line per episode.

    python -m baselines.run                                    # whole lineup, 3 repeats
    python -m baselines.run --models gpt-oss-20b --tasks easy_user_count --repeats 1

Resumable: (task, repeat) pairs already in baselines/results/<model>.jsonl are
skipped. A model whose free quota runs out is stopped, not scored, and an
episode that fails for infrastructure reasons is not recorded.
"""

import argparse
import datetime
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.env import SQLAnalystEnv  # noqa: E402
from environment.tasks import TASKS  # noqa: E402
from inference import MAX_STEPS as MAX_TURNS, SYSTEM_PROMPT, parse_model_response  # noqa: E402

from .providers import LINEUP, InfraError, QuotaExhausted, new_chat  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def run_episode(chat, task_id):
    """Play one episode exactly as inference.py does, and record everything."""
    env = SQLAnalystEnv()
    observation = env.reset(task_id=task_id)
    messages = [
        {"role": "system", "content": "You are a SQL expert. Respond only with valid JSON."},
        {"role": "user", "content": SYSTEM_PROMPT.format(
            schema_info=observation.schema_info,
            current_question=observation.current_question,
            last_query_result=observation.last_query_result,
            error_section="",
        )},
    ]
    record = {"actions": [], "parse_errors": 0, "tokens_in": 0, "tokens_out": 0,
              "served_model": "", "submitted": None, "destructive": False}
    info, done, turns = {}, False, 0
    started = time.monotonic()
    try:
        while not done and turns < MAX_TURNS:
            turns += 1
            text, usage = chat.reply(messages)
            record["tokens_in"] += usage["tokens_in"]
            record["tokens_out"] += usage["tokens_out"]
            record["served_model"] = usage["served_model"] or record["served_model"]
            messages.append({"role": "assistant", "content": text or "(empty response)"})

            action = parse_model_response(text)
            if action is None:
                record["parse_errors"] += 1
                record["actions"].append({"turn": turns, "kind": "parse_error", "text": text[:300]})
                messages.append({"role": "user", "content": (
                    "Your previous response was not valid JSON. Reply with exactly one of "
                    '{"sql_query": "..."} or {"submit_answer": "..."} and nothing else.')})
                continue

            observation, reward, done, info = env.step(action)
            kind = "sql" if action.sql_query else "answer"
            record["actions"].append({"turn": turns, "kind": kind,
                                      "text": action.sql_query or action.submit_answer,
                                      "reward": reward.value})
            if kind == "answer":
                record["submitted"] = action.submit_answer
            if reward.value == -1.0:
                record["destructive"] = True
            outcome = observation.error_message or observation.last_query_result
            messages.append({"role": "user", "content": (
                f"RESULT OF YOUR LAST ACTION:\n{outcome}\n\n"
                'If this answers the question, respond with {"submit_answer": "..."}. '
                'Otherwise respond with another {"sql_query": "..."}. JSON only.')})
    finally:
        env.close()

    record.update({
        "turns": turns,
        "env_steps": info.get("step_count", 0),
        "success": bool(info.get("success")),
        "verified": bool(info.get("verified")),
        "final_score": info.get("final_score", 0.0),
        "total_reward": round(info.get("total_reward", 0.0), 4),
        "seconds": round(time.monotonic() - started, 1),
    })
    return record


def completed(path):
    if not path.exists():
        return set()
    return {(r["task_id"], r["repeat"]) for r in map(json.loads, path.read_text().splitlines())}


def run_model(spec, task_ids, repeats, results_dir=RESULTS, chat_factory=new_chat, log=print):
    """Run every missing episode for one model. Returns how many were recorded."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{spec.slug}.jsonl"
    done = completed(path)
    recorded = 0
    for repeat in range(1, repeats + 1):
        for task_id in task_ids:
            if (task_id, repeat) in done:
                continue
            try:
                record = run_episode(chat_factory(spec), task_id)
            except QuotaExhausted as exc:
                log(f"  {spec.slug}: free quota exhausted, stopping this model ({exc})")
                return recorded
            except InfraError as exc:
                log(f"  {spec.slug} {task_id} r{repeat}: not recorded, infrastructure error ({exc})")
                continue
            record = {"model": spec.slug, "provider": spec.provider, "model_id": spec.model_id,
                      "served_via": spec.served_via, "task_id": task_id, "repeat": repeat,
                      **record,
                      "finished_at": datetime.datetime.now().isoformat(timespec="seconds")}
            with path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            recorded += 1
            log(f"  {spec.slug:<18} {task_id:<30} r{repeat}  "
                f"{'solved' if record['success'] else 'failed':<6} "
                f"score={record['final_score']:.2f} steps={record['env_steps']}")
    return recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--models", help="comma-separated slugs (default: whole lineup)")
    parser.add_argument("--tasks", help="comma-separated task ids (default: all)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    specs = LINEUP
    if args.models:
        wanted = args.models.split(",")
        specs = [s for s in LINEUP if s.slug in wanted]
        unknown = set(wanted) - {s.slug for s in specs}
        if unknown:
            parser.error(f"unknown models: {sorted(unknown)}; known: {[s.slug for s in LINEUP]}")
    task_ids = args.tasks.split(",") if args.tasks else [t.task_id for t in TASKS]

    # One empty working directory for every Claude episode in this run, so the
    # CLI's session files land in a single place.
    workdir = tempfile.mkdtemp(prefix="sqlgym-claude-")
    for spec in specs:
        print(f"== {spec.slug} ({spec.served_via})", flush=True)
        run_model(spec, task_ids, args.repeats,
                  chat_factory=lambda s: new_chat(s, workdir),
                  log=lambda line: print(line, flush=True))
    if any(s.provider == "claude-cli" for s in specs):
        print(f"Claude CLI working directory: {workdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
