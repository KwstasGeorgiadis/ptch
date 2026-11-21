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
    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "runs"
        / "professor"
        / "encoder_cycles_3.pt"
    )

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = Architecture().to("mps")
    model.trm.rule_cycles = saved["rule_cycles"]
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    print(f"Loaded: {checkpoint}", flush=True)

    # Show one saved validation example from each rule family.
    family = "add_one"  # ή "recolour" ή "rotate"
    example = 8          # διάλεξε από 1 έως 10

    episodes = [
        episode
        for episode in saved["evaluation_episodes"]
        if episode["family"] == family
    ]
    episode = episodes[example - 1]

    fig = plot_examples(
        model,
        episode["support"],
        [episode["query"]],
        first_number=example,
        set_name=family,
    )

    fig.suptitle(f"Validation — {family}", fontsize=16)
    output_file = checkpoint.parent / f"preview_{family}_{example}.png"
    fig.savefig(output_file, dpi=150, facecolor="white")
    print(f"Saved: {output_file}", flush=True)

    plt.show(block=True)


if __name__ == "__main__":
    main()
