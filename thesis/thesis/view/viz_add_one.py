"""Inspect saved add-one predictions. This script does not train the model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from ..evaluation import decode_prediction
from ..model import Architecture
from .visualization import ARC_COLOURS


ERROR_COLOURS = ListedColormap([
    "#FFFFFF", "#E4E4E7", "#E96572", "#F5CA73", "#B8A2DF",
])
ERROR_SYMBOLS = ["", "·", "X", "M", "E"]


def compare_grids(expected, predicted):
    # Compare the actual decoded output, including missing and extra cells.
    height = max(expected.shape[0], predicted.shape[0])
    width = max(expected.shape[1], predicted.shape[1])
    target = np.full((height, width), -1)
    output = np.full((height, width), -1)
    target[:expected.shape[0], :expected.shape[1]] = expected
    output[:predicted.shape[0], :predicted.shape[1]] = predicted

    both = (target >= 0) & (output >= 0)
    errors = np.zeros((height, width), dtype=int)
    errors[both & (target == output)] = 1
    errors[both & (target != output)] = 2
    errors[(target >= 0) & (output < 0)] = 3
    errors[(target < 0) & (output >= 0)] = 4
    return errors


def draw_grid(ax, grid, title, height, width, error_map=False):
    ax.imshow(
        grid, cmap=ERROR_COLOURS if error_map else ARC_COLOURS,
        vmin=0, vmax=4 if error_map else 9, interpolation="nearest",
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xticks(range(0, width, 1 if width <= 12 else 5))
    ax.set_yticks(range(0, height, 1 if height <= 12 else 5))
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#777777", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(labelsize=9)

    if max(height, width) <= 12:
        for row, column in np.ndindex(grid.shape):
            value = int(grid[row, column])
            label = ERROR_SYMBOLS[value] if error_map else str(value)
            colour = "white" if not error_map and value in (0, 1, 9) else "black"
            ax.text(column, row, label, ha="center", va="center",
                    color=colour, fontsize=11)


@torch.no_grad()
def plot_examples(model, support, queries, first_number, set_name):
    fig, axes = plt.subplots(
        len(queries), 4, figsize=(15, 3.6 * len(queries) + 1.2),
        squeeze=False,
    )
    model.eval()
    for row, query in enumerate(queries):
        prediction = model(support, query["input"])
        source = np.array(query["input"])
        expected = np.array(query["output"])
        predicted = np.array(decode_prediction(prediction))
        errors = compare_grids(expected, predicted)
        wrong, missing, extra = [int((errors == code).sum()) for code in (2, 3, 4)]
        number = first_number + row
        size_ok = expected.shape == predicted.shape
        exact = size_ok and wrong == 0

        height = max(source.shape[0], errors.shape[0])
        width = max(source.shape[1], errors.shape[1])
        titles = [
            f"{set_name} {number}: input\n{source.shape[0]}×{source.shape[1]}",
            f"Correct output\n{expected.shape[0]}×{expected.shape[1]}",
            f"Model prediction\n{predicted.shape[0]}×{predicted.shape[1]}"
            + (" · size correct" if size_ok else " · WRONG SIZE"),
            f"Errors\n{wrong} wrong · {missing} missing · {extra} extra",
        ]
        for column, grid in enumerate((source, expected, predicted, errors)):
            draw_grid(axes[row, column], grid, titles[column], height, width,
                      error_map=column == 3)

        print(f"{set_name} {number}: predicted {predicted.shape}, "
              f"expected {expected.shape} | wrong colours {wrong} | "
              f"missing cells {missing} | extra cells {extra} | exact {exact}")

    fig.legend(
        handles=[Patch(facecolor=ERROR_COLOURS(i), label=label)
                 for i, label in enumerate([
                     "· Correct", "X Wrong colour", "M Missing cell", "E Extra cell",
                 ], start=1)],
        loc="lower center", ncol=4, frameon=False, fontsize=12,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=2.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("unseen", "train"), default="unseen")
    parser.add_argument("--start", type=int, default=1, help="First example number (1-based)")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(__file__).resolve().parents[2] / "runs" / "double_canvas_size_only.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = "validation_queries" if args.set == "unseen" else "queries"
    pool = saved[key]
    if not 1 <= args.start <= len(pool) or args.count < 1:
        parser.error(f"Use --start 1 to {len(pool)} and --count at least 1.")
    queries = pool[args.start - 1:args.start - 1 + args.count]

    model = Architecture().to(args.device)
    model.load_state_dict(saved["model_state_dict"])
    print(f"Loaded {checkpoint} | trained on {len(saved['queries'])} examples")
    fig = plot_examples(model, saved["support"], queries, args.start, args.set.capitalize())
    fig.suptitle(
        f"Saved add-one model · {len(saved['queries'])} training examples\n"
        f"{args.set.capitalize()} examples {args.start}–{args.start + len(queries) - 1}",
        fontsize=16, y=0.99,
    )
    if args.save:
        fig.savefig(args.save, dpi=150, facecolor="white")
        print(f"Saved figure: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
