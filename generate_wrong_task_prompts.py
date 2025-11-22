"""Builds copy-pastable prompts for tasks the model missed."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "assets" / "arc_agi_v1_public_eval"
EVAL_ROOT = BASE_DIR / "assets" / "arc-agi" / "data" / "evaluation"

INTRO_TEXT = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. There is a pattern "
    "or transformation in the training examples that maps the input to the output. "
    "Now the test input differs slightly from the train input. Ideally a "
    "generalised transformation should have been applied to the test input to "
    "generate the test output. But the puzzle solver got confused and made a "
    "mistake. Figure out the mistake and generate the correct test output by "
    "correcting the mistake and applying the generalised transformation.\n\n"
    "Respond in the format of the output grids"
)


def load_entries(path: pathlib.Path) -> List[Dict[str, Any]]:
    payload = json.load(path.open())
    return payload if isinstance(payload, list) else [payload]


def attempt_correct(entry: Dict[str, Any]) -> bool:
    # Treat any attempt_n with correct=True as success.
    for key, val in entry.items():
        if isinstance(key, str) and key.startswith("attempt_"):
            attempt = val or {}
            if bool(attempt.get("correct")):
                return True
    return False


def gather_wrong_task_files(model_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(model_dir.glob("*.json")):
        try:
            entries = load_entries(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if not entries:
            continue
        if any(attempt_correct(e) for e in entries):
            continue
        yield path


def format_grid(grid: Any) -> str:
    if isinstance(grid, list) and grid and all(isinstance(r, list) for r in grid):
        row_strs = [", ".join(str(val) for val in row) for row in grid]
        inner = ",\n ".join(f"[{row}]" for row in row_strs)
        return "[\n " + inner + "\n]"
    return json.dumps(grid)


def preferred_attempts(preferred: int) -> List[int]:
    other = 1 if preferred == 2 else 2
    return [preferred, other]


def pick_wrong_answer(entry: Dict[str, Any], preferred: int) -> Any:
    for idx in preferred_attempts(preferred):
        attempt = entry.get(f"attempt_{idx}") or {}
        if "answer" in attempt and attempt["answer"] is not None:
            return attempt["answer"]
    raise ValueError("Missing answer payload for entry.")


def pick_pair_index(entry: Dict[str, Any]) -> int:
    for idx in (2, 1):
        meta = (entry.get(f"attempt_{idx}") or {}).get("metadata", {}) or {}
        pair_index = meta.get("pair_index")
        if pair_index is not None:
            return int(pair_index)
    return 0


def build_prompt(
    train_examples: List[Dict[str, Any]],
    test_input: Any,
    wrong_output: Any,
) -> str:
    lines: List[str] = [INTRO_TEXT, "", "--Training Examples--"]

    for idx, example in enumerate(train_examples):
        lines.extend(
            [
                f"--Example {idx}-- ",
                "",
                "INPUT: ",
                "",
                format_grid(example["input"]),
                "",
                "OUTPUT: ",
                "",
                format_grid(example["output"]),
                "",
            ]
        )

    lines.extend(
        [
            "--End of Training Examples--",
            "",
            "--Test Example--",
            "",
            " INPUT: ",
            "",
            format_grid(test_input),
            "",
            "WRONG OUTPUT:",
            "",
            format_grid(wrong_output),
            "",
            "--End of Test Examples--",
            "",
            "Your response:",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate codeblock prompts for every task where the model missed both "
            "attempts."
        )
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="gemini-3-pro-preview",
        help="Folder name within assets/arc_agi_v1_public_eval/.",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Preferred attempt to treat as the WRONG OUTPUT. Falls back to the other "
            "attempt if missing."
        ),
    )
    args = parser.parse_args()

    model_path = MODEL_ROOT / args.model_dir
    if not model_path.is_dir():
        parser.error(f"{model_path} is not a directory")
    if not EVAL_ROOT.is_dir():
        parser.error(f"Missing evaluation directory at {EVAL_ROOT}")

    any_output = False
    for wrong_path in gather_wrong_task_files(model_path):
        task_id = wrong_path.stem
        gold_path = EVAL_ROOT / f"{task_id}.json"
        if not gold_path.exists():
            print(f"Missing gold task file for {task_id}, skipping.", file=sys.stderr)
            continue

        gold = json.load(gold_path.open())
        train_examples = gold.get("train") or []
        test_examples = gold.get("test") or []
        if not test_examples:
            print(f"No test examples found for {task_id}, skipping.", file=sys.stderr)
            continue

        entries = load_entries(wrong_path)
        for entry_idx, entry in enumerate(entries):
            pair_index = pick_pair_index(entry)
            if pair_index >= len(test_examples):
                print(
                    f"Pair index {pair_index} out of range for {task_id}, skipping.",
                    file=sys.stderr,
                )
                continue
            test_input = test_examples[pair_index].get("input")
            try:
                wrong_output = pick_wrong_answer(entry, preferred=args.attempt)
            except ValueError as exc:
                print(f"Skipping {task_id} #{entry_idx}: {exc}", file=sys.stderr)
                continue

            prompt = build_prompt(train_examples, test_input, wrong_output)
            header = f"### Task {task_id} (pair {pair_index}) ###"
            print(header)
            print("```")
            print(prompt)
            print("```")
            print()
            any_output = True

    if not any_output:
        print("No wrong tasks found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
