import argparse
import json
import pathlib
import subprocess
import sys

ASSETS_DIR = pathlib.Path(__file__).resolve().parent / "../assets"
DATASET_DIR = ASSETS_DIR / "arc_agi_v1_public_eval"
CODE_DIR = ASSETS_DIR / "arc-agi"


def ensure_assets_downloaded() -> None:
    """Clone ARC dataset/code repos into assets/ when missing."""
    ASSETS_DIR.mkdir(exist_ok=True)
    clones = [
        (
            DATASET_DIR,
            [
                "git",
                "clone",
                "https://huggingface.co/datasets/arcprize/arc_agi_v1_public_eval/",
                str(DATASET_DIR),
            ],
        ),
        (
            CODE_DIR,
            ["git", "clone", "https://github.com/fchollet/arc-agi.git", str(CODE_DIR)],
        ),
    ]
    for path, command in clones:
        if path.exists():
            continue
        subprocess.run(command, check=True, cwd=ASSETS_DIR)


def task_correct(path: pathlib.Path) -> bool:
    """Return True if any attempt in the file is marked correct."""
    with path.open() as f:
        payload = json.load(f)

    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        attempt_1 = entry.get("attempt_1", {}) or {}
        attempt_2 = entry.get("attempt_2", {}) or {}
        if attempt_1.get("correct") or attempt_2.get("correct"):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List task ids where the model missed both attempts."
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="gemini-3-pro-preview",
        help=(
            "Final folder name within assets/arc_agi_v1_public_eval/. "
            "Defaults to gemini-3-pro-preview."
        ),
    )
    args = parser.parse_args()

    ensure_assets_downloaded()

    base_dir = DATASET_DIR
    llm_dir: pathlib.Path = base_dir / args.model_dir
    if not llm_dir.is_dir():
        parser.error(f"{llm_dir} is not a directory")

    wrong_ids = []
    total = 0

    for path in sorted(llm_dir.glob("*.json")):
        total += 1
        try:
            if not task_correct(path):
                wrong_ids.append(path.stem)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)

    for task_id in wrong_ids:
        print(task_id)

    print(f"Incorrect: {len(wrong_ids)} of {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
