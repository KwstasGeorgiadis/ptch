import random

import torch

from .data import sample_episode
from .losses import compute_loss


def decode_prediction(prediction):
    height = prediction["height_logits"].argmax(dim=1).item() + 1
    width = prediction["width_logits"].argmax(dim=1).item() + 1

    colours = prediction["colour_logits"].argmax(dim=1)[0]

    return colours[:height, :width].detach().cpu().tolist()


@torch.no_grad()
def evaluate_episodes(model, episodes):
    # Each episode is (support_pairs, query_pair).
    was_training = model.training
    model.eval()

    correct_cells = 0
    total_cells = 0
    correct_sizes = 0
    correct_grids = 0

    losses = {"colour": 0.0, "height": 0.0, "width": 0.0}
    details = []

    try:
        for support, query in episodes:
            prediction = model(support, query["input"])
            output = decode_prediction(prediction)

            _, parts = compute_loss(
                prediction, query["output"]
            )

            for name in losses:
                losses[name] += parts[name].item()

            target = torch.tensor(
                query["output"],
                device=prediction["colour_logits"].device,
            )

            height, width = target.shape
            colours = prediction["colour_logits"].argmax(dim=1)[0]

            cells = (
                colours[:height, :width] == target
            ).sum().item()

            predicted_size = (len(output), len(output[0]))
            size_ok = predicted_size == (height, width)
            exact_ok = output == query["output"]

            correct_cells += cells
            total_cells += target.numel()
            correct_sizes += int(size_ok)
            correct_grids += int(exact_ok)

            details.append({
                "predicted_size": predicted_size,
                "target_size": (height, width),
                "wrong_cells": target.numel() - int(cells),
                "total_cells": target.numel(),
                "exact": exact_ok,
            })
    finally:
        model.train(was_training)

    count = len(details)

    mean_losses = {
        name: loss / count
        for name, loss in losses.items()
    }

    return {
        "episodes": count,
        "target_canvas_cell_accuracy": correct_cells / total_cells,
        "size_accuracy": correct_sizes / count,
        "exact_grid_accuracy": correct_grids / count,
        "correct_sizes": correct_sizes,
        "correct_grids": correct_grids,
        "mean_loss": sum(mean_losses.values()),
        "losses": mean_losses,
        "details": details,
    }


def evaluate(model, index, task_ids, episodes=100, seed=123):
    rng = random.Random(seed)
    examples = []

    for _ in range(episodes):
        _, support, query = sample_episode(
            index, task_ids, rng
        )
        examples.append((support, query))

    return evaluate_episodes(model, examples)


def print_report(report):
    print(
        f"Correct cells: "
        f"{report['target_canvas_cell_accuracy']:.1%}"
    )
    print(
        f"Correct output sizes: "
        f"{report['size_accuracy']:.1%}"
    )
    print(
        f"Entire grids correct: "
        f"{report['exact_grid_accuracy']:.1%}"
    )