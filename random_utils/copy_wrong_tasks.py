"""
Copy incorrect ARC tasks into qwerty/, skipping specific ids.

Uses the same logic as list_wrong_tasks.py to identify incorrect tasks in
assets/arc_agi_v1_public_eval/gemini-3-pro-preview.
"""

import pathlib
import shutil
import sys
from typing import Iterable, Set

import list_wrong_tasks

MODEL_DIRNAME = "gemini-3-pro-preview"
DEST_DIRNAME = "qwerty"
SKIP_IDS: Set[str] = {
    "0d87d2a6",
    "1c56ad9f",
    "4e45f183",
    "8fbca751",
    "b942fd60",
    "dc2e9a9d",
    "184a9768",
    "50f325b5",
    "7e02026e",
    "ac0c5833",
    "d94c3b52",
    "c92b942c",
    "0934a4d8",
    "1acc24af",
    "256b0a75",
    "3ed85e70",
    "712bf12e",
    "73ccf9c2",
    "79fb03f4",
    "7bb29440",
    "7d419a02",
    "817e6c09",
    "8cb8642d",
    "9caba7c3",
    "bb52a14b",
    "c6e1b8da",
    "e1d2900e",
    "e619ca6e",
    "e681b708",
    "e6de6e8f",
    "f8be4b64",
    "f9d67f8b",
}


def iter_wrong_tasks(
    source_dir: pathlib.Path, skip_ids: Set[str], skipped: Set[str]
) -> Iterable[pathlib.Path]:
    """Yield incorrect task JSON files while tracking explicitly skipped ids."""
    for path in sorted(source_dir.glob("*.json")):
        if path.stem in skip_ids:
            skipped.add(path.stem)
            continue

        try:
            if not list_wrong_tasks.task_correct(path):
                yield path
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)


def main() -> int:
    list_wrong_tasks.ensure_assets_downloaded()

    source_dir = list_wrong_tasks.DATASET_DIR / MODEL_DIRNAME
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    dest_dir = pathlib.Path(__file__).resolve().parent.parent / DEST_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    skipped_ids: Set[str] = set()
    wrong_paths = list(iter_wrong_tasks(source_dir, SKIP_IDS, skipped_ids))

    copied = 0
    for path in wrong_paths:
        target = dest_dir / path.name
        shutil.copy2(path, target)
        copied += 1

    print(f"Copied {copied} incorrect task file(s) to {dest_dir}")
    if skipped_ids:
        print(
            f"Explicitly skipped {len(skipped_ids)} task id(s): {', '.join(sorted(skipped_ids))}"
        )

    missing_skips = SKIP_IDS - skipped_ids
    if missing_skips:
        print(f"Skip list ids not found in source: {', '.join(sorted(missing_skips))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
