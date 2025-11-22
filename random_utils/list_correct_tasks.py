"""
List task ids that contain at least one correct attempt.

Usage:
    python random_utils/list_correct_tasks.py path/to/run_dir
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List


def load_entries(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a JSON file that may contain either one object or a list."""
    payload = json.load(path.open())
    return payload if isinstance(payload, list) else [payload]


def entry_has_correct(entry: Dict[str, Any]) -> bool:
    """Return True if any attempt_* block (or the entry itself) is marked correct."""
    if entry.get("correct") is True:
        return True
    for key, value in entry.items():
        if not key.startswith("attempt_"):
            continue
        if isinstance(value, dict) and value.get("correct"):
            return True
    return False


def file_has_correct(path: pathlib.Path) -> bool:
    entries = load_entries(path)
    return any(entry_has_correct(entry) for entry in entries)


def iter_json_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield *.json files from the provided directory, sorted for stability."""
    yield from sorted(root.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List task ids with at least one correct attempt in the given directory."
    )
    parser.add_argument(
        "run_dir",
        help="Directory containing ARC attempt JSON files (e.g., gemini-rerun-samples-3).",
    )
    args = parser.parse_args()

    run_path = pathlib.Path(args.run_dir)
    if not run_path.is_dir():
        parser.error(f"{run_path} is not a directory")

    correct_ids: List[str] = []
    total = 0

    for path in iter_json_files(run_path):
        total += 1
        try:
            if file_has_correct(path):
                correct_ids.append(path.stem)
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)

    for task_id in correct_ids:
        print(task_id)

    print(f"Correct: {len(correct_ids)} of {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
