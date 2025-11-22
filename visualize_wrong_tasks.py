import argparse
import json
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


BASE = pathlib.Path(__file__).resolve().parent
MODEL_ROOT = BASE / "assets" / "arc_agi_v1_public_eval"
EVAL_ROOT = BASE / "assets" / "arc-agi" / "data" / "evaluation"

DEFAULT_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 aqua
    "#B10DC9",  # 9 purple
]


def load_entries(path: pathlib.Path) -> List[Dict[str, Any]]:
    payload = json.load(path.open())
    return payload if isinstance(payload, list) else [payload]


def attempt_correct(entry: Dict[str, Any]) -> bool:
    attempt_1 = entry.get("attempt_1", {}) or {}
    attempt_2 = entry.get("attempt_2", {}) or {}
    return bool(attempt_1.get("correct")) or bool(attempt_2.get("correct"))


def gather_wrong_tasks(model_dir: pathlib.Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for path in sorted(model_dir.glob("*.json")):
        entries = load_entries(path)
        if not entries:
            continue
        entry = entries[0]
        if attempt_correct(entry):
            continue
        yield path.stem, entry


def grid_to_array(grid: Any) -> Optional[np.ndarray]:
    if isinstance(grid, list) and grid and all(isinstance(r, list) for r in grid):
        try:
            return np.array(grid, dtype=int)
        except Exception:
            return None
    return None


def load_gold(task_id: str, pair_index: int = 0) -> Optional[np.ndarray]:
    path = EVAL_ROOT / f"{task_id}.json"
    if not path.exists():
        return None
    data = json.load(path.open())
    tests = data.get("test", [])
    if not tests:
        return None
    if pair_index < 0 or pair_index >= len(tests):
        return None
    return grid_to_array(tests[pair_index].get("output"))


def build_dataset(model_dir: pathlib.Path) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for task_id, entry in gather_wrong_tasks(model_dir):
        attempt_1 = entry.get("attempt_1", {}) or {}
        attempt_2 = entry.get("attempt_2", {}) or {}
        dataset.append(
            {
                "task_id": task_id,
                "attempt_1": grid_to_array(attempt_1.get("answer")),
                "attempt_1_correct": bool(attempt_1.get("correct")),
                "attempt_2": grid_to_array(attempt_2.get("answer")),
                "attempt_2_correct": bool(attempt_2.get("correct")),
                "gold": load_gold(task_id),
            }
        )
    return dataset


def draw(ax, arr: Optional[np.ndarray], title: str, cmap):
    ax.clear()
    if arr is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=len(DEFAULT_COLORS) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return im


def visualize(dataset: List[Dict[str, Any]]):
    if not dataset:
        print("No wrong tasks found.")
        return

    cmap = ListedColormap(DEFAULT_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    status_text = fig.text(
        0.5, 0.02, "Use left/right arrow keys to navigate. Close window to exit.",
        ha="center", fontsize=10
    )

    state = {"idx": 0}

    def redraw():
        item = dataset[state["idx"]]
        task_id = item["task_id"]
        fig.suptitle(f"Task {task_id} ({state['idx']+1}/{len(dataset)})", fontsize=14)
        draw(
            axes[0],
            item["attempt_1"],
            f"attempt_1 ({'correct' if item['attempt_1_correct'] else 'wrong'})",
            cmap,
        )
        draw(
            axes[1],
            item["attempt_2"],
            f"attempt_2 ({'correct' if item['attempt_2_correct'] else 'wrong'})",
            cmap,
        )
        draw(axes[2], item["gold"], "gold output", cmap)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["idx"] = (state["idx"] + 1) % len(dataset)
            redraw()
        elif event.key == "left":
            state["idx"] = (state["idx"] - 1) % len(dataset)
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    # Silence linters about unused vars
    _ = status_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Matplotlib viewer for wrong tasks: attempt_1, attempt_2, gold."
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

    model_path = MODEL_ROOT / args.model_dir
    if not model_path.is_dir():
        parser.error(f"{model_path} is not a directory")
    if not EVAL_ROOT.is_dir():
        parser.error(f"Missing evaluation directory at {EVAL_ROOT}")

    dataset = build_dataset(model_path)
    visualize(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
