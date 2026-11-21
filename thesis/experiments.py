import torch

from .evaluation import evaluate_episodes
from .model import Architecture
from .training import training_step


def fit_fixed_examples(
    support,
    queries,
    steps=1000,
    device="mps",
    lower_lr_at=None,
):
    torch.manual_seed(0)

    model = Architecture().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    examples = [
        (support, query)
        for query in queries
    ]

    for step in range(steps + 1):
        if step > 0:
            if step == lower_lr_at:
                for group in optimizer.param_groups:
                    group["lr"] = 1e-4

            query = queries[(step - 1) % len(queries)]

            training_step(
                model, optimizer, support, query
            )

        if step % 100 == 0 or step == steps:
            report = evaluate_episodes(model, examples)
            losses = report["losses"]

            print(
                f"Step {step:4d} | "
                f"exact {report['correct_grids']}/{len(queries)} | "
                f"sizes {report['correct_sizes']}/{len(queries)} | "
                f"cells {report['target_canvas_cell_accuracy']:.1%} | "
                f"colour loss {losses['colour']:.3f} | "
                f"height loss {losses['height']:.3f} | "
                f"width loss {losses['width']:.3f}",
                flush=True,
            )

            if report["correct_grids"] == len(queries):
                print("All fixed examples are correct.")
                break

    for number, detail in enumerate(report["details"], 1):
        print(
            f"Example {number}: "
            f"size {detail['predicted_size']} "
            f"vs target {detail['target_size']} | "
            f"wrong cells {detail['wrong_cells']}/{detail['total_cells']} | "
            f"exact {detail['exact']}"
        )

    return model, report, step