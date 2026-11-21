import random
from pathlib import Path

import torch

from thesis.model import Architecture
from thesis.synthetic import make_synthetic_episode
from thesis.training import training_step
from thesis.evaluation import evaluate_episodes


STEPS = 2000
CHECK_EVERY = 200
VALIDATION_EXAMPLES = 100
LEARNING_RATE = 1e-4
DEVICE = "mps"

SAVE_FILE = Path(__file__).resolve().parents[1] / "runs" / "fresh_recolour.pt"


def new_query(rng, used_inputs):
    """Generate an input that has not appeared in either set."""
    while True:
        _, query = make_synthetic_episode(
            "recolour",
            rng,
            source=1,
            target=2,
        )

        key = tuple(map(tuple, query["input"]))

        if key not in used_inputs:
            used_inputs.add(key)
            return query


def check_validation(model, support, validation, step):
    episodes = [(support, query) for query in validation]
    report = evaluate_episodes(model, episodes)

    count = report["episodes"]

    print(
        f"Step {step:4d} | UNSEEN examples | "
        f"grids {report['correct_grids']}/{count} | "
        f"sizes {report['correct_sizes']}/{count} | "
        f"cells in target area "
        f"{report['target_canvas_cell_accuracy']:.1%} | "
        f"colour loss {report['losses']['colour']:.3f} | "
        f"height loss {report['losses']['height']:.3f} | "
        f"width loss {report['losses']['width']:.3f}",
        flush=True,
    )

    return report


def main():
    torch.manual_seed(0)

    # Keep these three solved examples fixed throughout this experiment.
    support, _ = make_synthetic_episode(
        "recolour",
        random.Random(0),
        source=1,
        target=2,
    )

    used_inputs = {
        tuple(map(tuple, pair["input"]))
        for pair in support
    }

    # Generate validation first. Its inputs are then excluded from training.
    validation_rng = random.Random(123)
    validation = [
        new_query(validation_rng, used_inputs)
        for _ in range(VALIDATION_EXAMPLES)
    ]

    # Start from fresh weights, without loading the previous checkpoint.
    model = Architecture().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    training_rng = random.Random(456)
    training_queries = []
    accumulated_loss = 0.0
    accumulated_steps = 0

    print(
        "Fresh model | Replace blue (1) with red (2)\n"
        "New training input every update | 100 fixed validation examples",
        flush=True,
    )

    check_validation(model, support, validation, step=0)

    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, STEPS + 1):
        # Generate a genuinely new training input.
        query = new_query(training_rng, used_inputs)
        training_queries.append(query)

        # Your existing function trains BOTH colours and output dimensions.
        loss = training_step(
            model,
            optimizer,
            support,
            query,
        )

        accumulated_loss += loss
        accumulated_steps += 1

        if step % CHECK_EVERY == 0 or step == STEPS:
            print(
                f"\nMean training loss over the last "
                f"{accumulated_steps} updates: "
                f"{accumulated_loss / accumulated_steps:.4f}",
                flush=True,
            )

            report = check_validation(
                model,
                support,
                validation,
                step,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "family": "recolour",
                    "parameters": {"source": 1, "target": 2, "turns": 1},
                    "support": support,
                    "queries": training_queries,
                    "validation_queries": validation,
                    "validation_report": report,
                    "steps": step,
                },
                SAVE_FILE,
            )

            print(f"Saved: {SAVE_FILE}", flush=True)

            accumulated_loss = 0.0
            accumulated_steps = 0


if __name__ == "__main__":
    main()