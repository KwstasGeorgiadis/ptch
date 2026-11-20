import random
from pathlib import Path

import torch

from .evaluation import evaluate_episodes
from .model import Architecture
from .synthetic import make_synthetic_episode
from .training import training_step


START_FILE = Path("runs/synthetic_examples.pt")
OUTPUT_FILE = Path("runs/add_one.pt")

DEVICE = "mps"
MAX_EXAMPLES = 20
MAX_PASSES = 200
CHECK_EVERY = 10
VALIDATION_EXAMPLES = 100
LEARNING_RATE = 1e-4


def new_queries(count, rng, used_inputs, family, parameters):
    queries = []

    while len(queries) < count:
        _, query = make_synthetic_episode(
            family, rng, **parameters
        )

        key = tuple(map(tuple, query["input"]))

        if key not in used_inputs:
            used_inputs.add(key)
            queries.append(query)

    return queries


def print_result(label, report):
    count = report["episodes"]

    print(
        f"{label} | grids {report['correct_grids']}/{count} | "
        f"sizes {report['correct_sizes']}/{count} | "
        f"cells {report['target_canvas_cell_accuracy']:.1%} | "
        f"loss {report['mean_loss']:.4f}",
        flush=True,
    )


def main():
    saved = torch.load(
        START_FILE,
        map_location="cpu",
        weights_only=True,
    )

    support = saved["support"]
    initial_queries = saved["queries"]
    family = saved["family"]
    parameters = saved["parameters"]

    model = Architecture().to(DEVICE)
    model.load_state_dict(saved["model_state_dict"])

    # The original checkpoint did not save its optimizer.
    # Create one now and keep it throughout all stages.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    used_inputs = {
        tuple(map(tuple, pair["input"]))
        for pair in support + initial_queries
    }

    # Fixed validation examples, never used for training.
    validation = new_queries(
        VALIDATION_EXAMPLES,
        random.Random(123),
        used_inputs,
        family,
        parameters,
    )

    # Generate the examples that will gradually join training.
    pool = initial_queries + new_queries(
        MAX_EXAMPLES - len(initial_queries),
        random.Random(456),
        used_inputs,
        family,
        parameters,
    )

    validation_episodes = [
        (support, query) for query in validation
    ]

    order_rng = random.Random(789)
    history = []
    updates = 0

    print("Continuing the saved model.")

    print_result(
        "Unseen before training",
        evaluate_episodes(model, validation_episodes),
    )

    for size in range(len(initial_queries), MAX_EXAMPLES + 1):
        queries = pool[:size]
        training_episodes = [
            (support, query) for query in queries
        ]

        report = evaluate_episodes(model, training_episodes)
        print_result(
            f"\nTraining set {size}, before updates",
            report,
        )

        passes = 0

        for pass_number in range(1, MAX_PASSES + 1):
            if report["correct_grids"] == size:
                break

            # Train on every old example and the new example.
            for query in order_rng.sample(queries, len(queries)):
                training_step(
                    model, optimizer, support, query
                )
                updates += 1

            passes = pass_number

            if passes % CHECK_EVERY == 0 or passes == MAX_PASSES:
                report = evaluate_episodes(
                    model, training_episodes
                )

                print_result(
                    f"Training set {size}, pass {passes}",
                    report,
                )

        validation_report = evaluate_episodes(
            model, validation_episodes
        )
        print_result("Fixed unseen set", validation_report)

        history.append({
            "training_examples": size,
            "passes": passes,
            "updates_since_start": updates,
            "training": report,
            "validation": validation_report,
        })

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "family": family,
            "parameters": parameters,
            "support": support,
            "queries": queries,
            "validation_queries": validation,
            "query_pool": pool,
            "report": report,
            "validation_report": validation_report,
            "history": history,
            "updates_since_start": updates,
        }, OUTPUT_FILE)

        print(f"Saved: {OUTPUT_FILE.resolve()}", flush=True)

        if report["correct_grids"] != size:
            print(
                f"Stopped at {size} examples "
                f"after {MAX_PASSES} passes."
            )
            print(
                "This is a training-budget limit, "
                "not proof of the model's capacity limit."
            )
            break

        if size < MAX_EXAMPLES:
            print(
                "Current training set is perfect. "
                "Adding one example."
            )


if __name__ == "__main__":
    main()