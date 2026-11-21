import random
from pathlib import Path

import torch

from .experiments import fit_fixed_examples
from .synthetic import make_synthetic_episode

FAMILY = "recolour"  # "recolour" or "rotate" or 
SOURCE = 1
TARGET = 2
TURNS = 1           # Clockwise quarter-turns: 1, 2, or 3. grid turns 
STEPS = 1000
DEVICE = "mps"
OUTPUT_FILE = Path("runs/synthetic_examples.pt")


def main():
    rng = random.Random(0)

    support, first_query = make_synthetic_episode(
        FAMILY,
        rng,
        source=SOURCE,
        target=TARGET,
        turns=TURNS,
    )

    queries = [first_query]

    while len(queries) < 5:
        _, query = make_synthetic_episode(
            FAMILY,
            rng,
            source=SOURCE,
            target=TARGET,
            turns=TURNS,
        )

        if all(
            query["input"] != pair["input"]
            for pair in support + queries
        ):
            queries.append(query)

    model, report, completed_steps = fit_fixed_examples(
        support,
        queries,
        steps=STEPS,
        device=DEVICE,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "family": FAMILY,
        "parameters": {
            "source": SOURCE,
            "target": TARGET,
            "turns": TURNS,
        },
        "support": support,
        "queries": queries,
        "steps": completed_steps,
        "report": report,
    }, OUTPUT_FILE)

    print(f"Saved weights: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()