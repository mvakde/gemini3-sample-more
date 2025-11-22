"""Lightweight helper to visualize a 2D grid of integers.

Edit the GRID variable below, then run `python grid_visualizer.py` to open
a matplotlib window showing the grid with ARC-style colors.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Update this grid with your own values.
GRID = [
    [1, 3, 2],
    [9, 2, 3],
    [9, 2, 3],
    [1, 3, 2],
    [4, 4, 7],
    [4, 4, 7],
    [6, 6, 1],
    [6, 6, 6],
    [1, 6, 1],
]

# Color palette matches the existing ARC tooling in this repo.
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


def grid_to_array(grid):
    """Convert nested lists into a 2D numpy array of ints."""
    if (
        not isinstance(grid, list)
        or not grid
        or not all(isinstance(r, list) for r in grid)
    ):
        raise ValueError("GRID must be a non-empty list of lists")
    try:
        arr = np.asarray(grid, dtype=int)
    except Exception as exc:
        raise ValueError("GRID must contain only integers") from exc
    if arr.ndim != 2:
        raise ValueError("GRID must be 2D, e.g., [[0, 1], [2, 3]]")
    return arr


def visualize_grid(arr, title="Grid visualization"):
    cmap = ListedColormap(DEFAULT_COLORS)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=len(DEFAULT_COLORS) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(len(DEFAULT_COLORS)))
    plt.tight_layout()
    plt.show()


def main():
    arr = grid_to_array(GRID)
    visualize_grid(arr)


if __name__ == "__main__":
    main()
