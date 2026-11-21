import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import torch

from ..evaluation import decode_prediction

ARC_COLOURS = ListedColormap([
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
])


@torch.no_grad()
def plot_predictions(model, episodes, show=True):
    # episodes: a list of (support, query) tuples.
    was_training = model.training
    model.eval()

    fig, axes = plt.subplots(
        len(episodes),
        3,
        figsize=(12, 3.5 * len(episodes)),
        squeeze=False,
    )

    try:
        for number, (support, query) in enumerate(episodes):
            prediction = model(support, query["input"])

            input_grid = np.array(query["input"])
            expected = np.array(query["output"])
            predicted = np.array(decode_prediction(prediction))

            grids = [input_grid, expected, predicted]

            titles = [
                f"Example {number + 1}: input",
                "Correct output",
                "Prediction",
            ]

            for column, (grid, title) in enumerate(zip(grids, titles)):
                ax = axes[number, column]
                height, width = grid.shape

                ax.imshow(
                    grid,
                    cmap=ARC_COLOURS,
                    vmin=0,
                    vmax=9,
                    interpolation="nearest",
                )

                ax.set_title(f"{title}\n{height}×{width}")

                ax.set_xticks(
                    range(0, width, 1 if width <= 15 else 5)
                )
                ax.set_yticks(
                    range(0, height, 1 if height <= 15 else 5)
                )
                ax.tick_params(labelsize=8)

                ax.set_xticks(
                    np.arange(-0.5, width, 1), minor=True
                )
                ax.set_yticks(
                    np.arange(-0.5, height, 1), minor=True
                )
                ax.grid(
                    which="minor",
                    color="#555555",
                    linewidth=0.5,
                )
                ax.tick_params(which="minor", length=0)

            if predicted.shape == expected.shape:
                wrong = np.argwhere(predicted != expected)

                axes[number, 2].set_title(
                    f"Prediction · {predicted.shape[0]}×{predicted.shape[1]}\n"
                    f"{len(wrong)} wrong cells"
                )

                for row, column in wrong:
                    for output_column in (1, 2):
                        axes[number, output_column].add_patch(
                            Rectangle(
                                (column - 0.5, row - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                            )
                        )
            else:
                axes[number, 2].set_title(
                    f"Wrong size: {predicted.shape[0]}×{predicted.shape[1]}\n"
                    f"Expected: {expected.shape[0]}×{expected.shape[1]}"
                )
    finally:
        model.train(was_training)

    fig.tight_layout()

    if show:
        plt.show()

    return fig