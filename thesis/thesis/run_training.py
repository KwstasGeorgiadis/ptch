from pathlib import Path

import torch

from .evaluation import evaluate, print_report
from .training import run_training

STEPS = 1000
DEVICE = "mps"
EVAL_EPISODES = 100
OUTPUT_FILE = Path("runs/arc_training.pt")


def main():
    model, index, val_ids = run_training(
        steps=STEPS,
        device=DEVICE,
    )

    report = evaluate(
        model,
        index,
        val_ids,
        episodes=EVAL_EPISODES,
    )

    print_report(report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "training_task_ids": sorted(set(index) - set(val_ids)),
        "validation_task_ids": val_ids,
        "steps": STEPS,
        "report": report,
    }, OUTPUT_FILE)

    print(f"Saved weights: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()