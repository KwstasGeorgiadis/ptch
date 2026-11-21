from pathlib import Path

import torch

from .data import index_task_files
from .evaluation import evaluate, evaluate_episodes, print_report
from .model import Architecture


MODEL_FILE = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "double_canvas_fresh.pt"
)

DEVICE = "mps"
EPISODES = 100


def main():
    saved = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = Architecture().to(DEVICE)
    model.load_state_dict(saved["model_state_dict"])

    print(f"Loaded: {MODEL_FILE}")

    if "support" in saved:
        # Synthetic run: use examples excluded from training.
        examples = [
            (saved["support"], query)
            for query in saved["validation_queries"]
        ]

        print(f"Checking {len(examples)} held-out synthetic examples.")

        report = evaluate_episodes(model, examples)

    else:
        # ARC run: use task IDs excluded from training.
        index = index_task_files()

        # Earlier training versions used different names for this key.
        if "validation_task_ids" in saved:
            validation_ids = saved["validation_task_ids"]
        else:
            validation_ids = saved["val_ids"]

        print("Checking held-out ARC tasks.")

        report = evaluate(
            model,
            index,
            validation_ids,
            episodes=EPISODES,
        )

    print_report(report)

    print(f"Colour loss: {report['losses']['colour']:.4f}")
    print(f"Height loss: {report['losses']['height']:.4f}")
    print(f"Width loss: {report['losses']['width']:.4f}")


if __name__ == "__main__":
    main()