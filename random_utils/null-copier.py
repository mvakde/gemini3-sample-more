from pathlib import Path
import json
import shutil

# Set these before running
SOURCE_DIR = Path("../gemini-rerun-samples-3")
DEST_DIR = Path("../gemini-rerun-samples-3-nulls")


def has_null_answer(node) -> bool:
    """Return True only when a dict has the key 'answer' explicitly set to None."""
    if isinstance(node, dict):
        if "answer" in node and node["answer"] is None:
            return True
        return any(has_null_answer(v) for v in node.values())
    if isinstance(node, list):
        return any(has_null_answer(v) for v in node)
    return False


def copy_null_tasks() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for path in SOURCE_DIR.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {path}: {exc}")
            continue

        if not has_null_answer(data):
            continue

        target = DEST_DIR / path.relative_to(SOURCE_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    print(f"Copied {copied} file(s) with null answers to {DEST_DIR}")


if __name__ == "__main__":
    copy_null_tasks()
