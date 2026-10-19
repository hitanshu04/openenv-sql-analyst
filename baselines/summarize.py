"""
Turn baselines/results/*.jsonl into markdown tables.

    python -m baselines.summarize                  # print the tables
    python -m baselines.summarize --write-readme   # also refresh README's results block
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.tasks import TASKS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- BASELINES:START -->", "<!-- BASELINES:END -->"

FAILURE_LABELS = {
    "correct_unchecked": "right answer without reading the data it needs",
    "near_miss": "near miss (within 10%)",
    "wrong": "wrong answer",
    "no_answer": "never answered",
    "destructive": "tried to modify the database",
}


def outcome(record):
    if record["destructive"]:
        return "destructive"
    if record["success"]:
        return "correct" if record["verified"] else "correct_unchecked"
    if record["submitted"] is None:
        return "no_answer"
    return "near_miss" if record["final_score"] == 0.5 else "wrong"


def load(results_dir=RESULTS):
    records = []
    for path in sorted(results_dir.glob("*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
    return records


def model_table(records):
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    expected = len(TASKS) * max(r["repeat"] for r in records)
    rows = []
    for model, rs in by_model.items():
        label = f"`{model}`" + (f" (partial: {len(rs)} of {expected} episodes)" if len(rs) < expected else "")
        solved = [r for r in rs if r["success"]]
        rate = len(solved) / len(rs)
        mean_score = sum(r["final_score"] for r in rs) / len(rs)
        steps = f"{sum(r['env_steps'] for r in solved) / len(solved):.1f}" if solved else "–"
        failures = Counter(outcome(r) for r in rs if outcome(r) != "correct")
        top = failures.most_common(1)
        failure = f"{FAILURE_LABELS[top[0][0]]} ({top[0][1]})" if top else "–"
        rows.append((rate, mean_score,
                     f"| {label} | {len(solved)}/{len(rs)} ({100 * rate:.0f}%) | {mean_score:.2f} "
                     f"| {steps} | {failures['correct_unchecked']} | {failure} "
                     f"| {rs[0]['served_via']} |"))
    rows.sort(key=lambda row: (-row[0], -row[1]))
    header = ("| Model | Solved | Mean score | Avg steps when solved | Right but unchecked "
              "| Most common failure | Served via |\n|---|---|---|---|---|---|---|")
    return "\n".join([header] + [row[2] for row in rows])


def task_table(records):
    models = sorted({r["model"] for r in records})
    cells = defaultdict(lambda: [0, 0])
    for r in records:
        cell = cells[(r["task_id"], r["model"])]
        cell[0] += r["success"]
        cell[1] += 1
    header = "| Task | " + " | ".join(f"`{m}`" for m in models) + " |\n|---|" + "---|" * len(models)
    lines = [header]
    for task in TASKS:
        row = [f"{cells[(task.task_id, m)][0]}/{cells[(task.task_id, m)][1]}"
               if cells[(task.task_id, m)][1] else "–" for m in models]
        lines.append(f"| `{task.task_id}` | " + " | ".join(row) + " |")
    return "\n".join(lines)


def report(records):
    dates = sorted(r["finished_at"][:10] for r in records)
    repeats = max(r["repeat"] for r in records)
    return "\n\n".join([
        f"{len(records)} episodes: {len({r['model'] for r in records})} models × {len(TASKS)} tasks "
        f"× up to {repeats} repeats, run {dates[0]} to {dates[-1]} with "
        "[`baselines/run.py`](baselines/run.py). Temperature 0, except Claude, whose CLI "
        "does not expose temperature. Raw episodes, including every query, are in "
        "[`baselines/results/`](baselines/results/).",
        model_table(records),
        "Per task (solved / runs):",
        task_table(records),
    ])


def write_readme(text, readme=README):
    content = readme.read_text()
    start, end = content.index(START) + len(START), content.index(END)
    readme.write_text(content[:start] + "\n" + text + "\n" + content[end:])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    records = load()
    if not records:
        print("no results yet")
        return 1
    text = report(records)
    print(text)
    if args.write_readme:
        write_readme(text)
        print(f"\nwrote {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
