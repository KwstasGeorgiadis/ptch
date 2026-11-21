import random
from pathlib import Path

import torch

from .data import index_task_files, load_task_pairs
from .experiments import fit_fixed_examples

TASK_ID = "007bbfb7"
STEPS = 4000
DEVICE = "mps"
OUTPUT_FILE = Path("runs/five_examples.pt")


def main():
    index = index_task_files()
    pairs = load_task_pairs(index[TASK_ID])

    support = pairs[:3]

    candidates = [
        pair for pair in pairs
        if all(
            pair["input"] != item["input"]
            for item in support
        )
        and len(pair["output"]) <= 9
        and len(pair["output"][0]) <= 9
    ]

    queries = random.Random(0).sample(candidates, 5)

    model, report, completed_steps = fit_fixed_examples(
        support,
        queries,
        steps=STEPS,
        device=DEVICE,
        lower_lr_at=2001,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "task_id": TASK_ID,
        "support": support,
        "queries": queries,
        "steps": completed_steps,
        "report": report,
    }, OUTPUT_FILE)

    print(f"Saved weights: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()