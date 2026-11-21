import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from .data import index_task_files, sample_episode, split_tasks
from .evaluation import evaluate, evaluate_episodes, print_report
from .losses import compute_loss
from .model import Architecture
from .synthetic import make_synthetic_episode


TASK = "all"  # "all", "fit", "compare", "split", "arc", "evaluate"
DEVICE = "mps"
SEED = 0

LEARNING_RATE = 1e-3
FIT_MAX_STEPS = 2000
FIT_CELL_TARGET = 0.99

STEPS = 1000              # Updates per encoder variant.
CHECK_EVERY = 200
VALIDATION_PER_FAMILY = 10

ADAPTATION_STEPS = 3
ADAPTATION_LR = 0.01

FAMILIES = ("recolour", "rotate", "double_canvas")

RUNS = Path(__file__).resolve().parents[1] / "runs" / "professor"
MODEL_FILE = RUNS / "encoder_cycles_3.pt"


def grid_key(grid):
    return tuple(map(tuple, grid))


def episode_inputs(episode):
    pairs = episode["support"] + [episode["query"]]
    return {grid_key(pair["input"]) for pair in pairs}


def compact(report):
    return {
        name: value
        for name, value in report.items()
        if name != "details"
    }


def show(label, report):
    count = report["episodes"]

    print(
        f"{label} | "
        f"cells {report['target_canvas_cell_accuracy']:.1%} | "
        f"sizes {report['correct_sizes']}/{count} | "
        f"grids {report['correct_grids']}/{count} | "
        f"loss {report['mean_loss']:.4f}",
        flush=True,
    )


def write_report(results):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"report_{TASK}.json"

    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Report saved: {path}", flush=True)


def new_model(rule_cycles=3):
    torch.manual_seed(SEED)
    model = Architecture().to(DEVICE)

    if not hasattr(model.trm, "rule_cycles"):
        raise RuntimeError(
            "Update RuleEncoder to use self.rule_cycles first."
        )

    model.trm.rule_cycles = rule_cycles
    return model


def make_optimizer(model, learning_rate=LEARNING_RATE):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )


def training_step(model, optimizer, support, query):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(support, query["input"])
    loss, _ = compute_loss(prediction, query["output"])

    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss became non-finite.")

    loss.backward()

    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )

    optimizer.step()
    return loss.item()


def measure(model, episodes):
    return evaluate_episodes(
        model,
        [
            (episode["support"], episode["query"])
            for episode in episodes
        ],
    )


def save_checkpoint(name, model, optimizer, episodes, split, step):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / name

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rule_cycles": model.trm.rule_cycles,
            "evaluation_episodes": episodes,
            "evaluation_split": split,
            "steps": step,
            "seed": SEED,
        },
        path,
    )

    print(f"Checkpoint saved: {path}", flush=True)


# Professor request 4: verify the actual task-level split.
def checked_task_split():
    index = index_task_files()
    train_ids, val_ids = split_tasks(index, seed=SEED)

    train_set = set(train_ids)
    val_set = set(val_ids)

    assert train_set.isdisjoint(val_set), "Task IDs overlap."
    assert train_set | val_set == set(index), "Some tasks are missing."

    train_files = {
        Path(path).resolve()
        for task_id in train_ids
        for path in index[task_id]
    }

    val_files = {
        Path(path).resolve()
        for task_id in val_ids
        for path in index[task_id]
    }

    # ARC and RE-ARC identify the original task by the filename stem.
    train_original_ids = {
        Path(path).stem
        for task_id in train_ids
        for path in index[task_id]
    }

    val_original_ids = {
        Path(path).stem
        for task_id in val_ids
        for path in index[task_id]
    }

    assert train_files.isdisjoint(val_files), "Source files overlap."
    assert train_original_ids.isdisjoint(val_original_ids), (
        "Examples from the same original task cross the split."
    )

    report = {
        "passed": True,
        "training_tasks": len(train_ids),
        "validation_tasks": len(val_ids),
        "training_files": len(train_files),
        "validation_files": len(val_files),
        "overlapping_original_task_ids": 0,
    }

    print(
        f"Task split PASS | "
        f"{len(train_ids)} training tasks | "
        f"{len(val_ids)} validation tasks | no overlap",
        flush=True,
    )

    return index, train_ids, val_ids, report


# Professor request 2: vary rule parameters between episodes.
# The model receives the support pairs, not these parameter labels.
def draw_episode(rng, family=None, forbidden=None):
    while True:
        chosen_family = family or rng.choice(FAMILIES)

        if chosen_family == "recolour":
            source, target = rng.sample(range(10), 2)
            parameters = {"source": source, "target": target}

        elif chosen_family == "rotate":
            parameters = {"turns": rng.choice((1, 2, 3))}

        elif chosen_family == "double_canvas":
            parameters = {"factor": rng.choice((2, 3))}

        else:
            raise ValueError(f"Unknown family: {chosen_family}")

        support, query = make_synthetic_episode(
            chosen_family, rng, **parameters
        )

        episode = {
            "family": chosen_family,
            "parameters": parameters,
            "support": support,
            "query": query,
        }

        if forbidden is None or episode_inputs(episode).isdisjoint(forbidden):
            return episode


def make_validation():
    rng = random.Random(123)
    episodes = []
    used_inputs = set()

    for family in FAMILIES:
        for _ in range(VALIDATION_PER_FAMILY):
            episode = draw_episode(rng, family, used_inputs)
            episodes.append(episode)
            used_inputs.update(episode_inputs(episode))

    return episodes


# Professor request 1: fit a tiny, fixed dataset with the full model.
def near_complete(report):
    return (
        report["target_canvas_cell_accuracy"] >= FIT_CELL_TARGET
        and report["correct_sizes"] == report["episodes"]
    )


def fit_tiny(family):
    rng = random.Random(17)
    parameters = (
        {"source": 1, "target": 2}
        if family == "recolour"
        else {}
    )

    support, first_query = make_synthetic_episode(
        family,
        rng,
        min_size=6,
        max_size=6,
        **parameters,
    )

    queries = [first_query]
    used = {
        grid_key(pair["input"])
        for pair in support + queries
    }

    while len(queries) < 5:
        _, query = make_synthetic_episode(
            family,
            rng,
            min_size=6,
            max_size=6,
            **parameters,
        )

        key = grid_key(query["input"])

        if key not in used:
            used.add(key)
            queries.append(query)

    episodes = [
        {
            "family": family,
            "parameters": parameters,
            "support": support,
            "query": query,
        }
        for query in queries
    ]

    model = new_model(rule_cycles=3)
    optimizer = make_optimizer(model)

    initial = measure(model, episodes)
    report = initial
    show(f"{family}: TRAINING examples, before training", report)

    step = 0

    while step < FIT_MAX_STEPS and not near_complete(report):
        query = queries[step % len(queries)]
        training_step(model, optimizer, support, query)
        step += 1

        if step % 100 == 0 or step == FIT_MAX_STEPS:
            report = measure(model, episodes)
            show(f"{family}: TRAINING examples, step {step}", report)

    save_checkpoint(
        f"tiny_{family}.pt",
        model,
        optimizer,
        episodes,
        split="training",
        step=step,
    )

    return {
        "passed": near_complete(report),
        "criterion": "At least 99% correct cells and all sizes correct.",
        "grid_size": [6, 6],
        "training_examples": 5,
        "updates": step,
        "initial": compact(initial),
        "final": compact(report),
    }


# Adjust only the rule vector using solved support pairs.
# This function never receives the query or its target.
def adapt_rule(model, support_pairs, steps=ADAPTATION_STEPS):
    was_training = model.training
    parameters = list(model.parameters())
    original_flags = [p.requires_grad for p in parameters]

    model.eval()

    try:
        for parameter in parameters:
            parameter.requires_grad_(False)

        with torch.no_grad():
            initial_rule = model.trm(support_pairs)

        with torch.enable_grad():
            rule = nn.Parameter(initial_rule.detach().clone())
            optimizer = torch.optim.Adam([rule], lr=ADAPTATION_LR)

            best_rule = rule.detach().clone()
            best_loss = float("inf")

            for step in range(steps + 1):
                optimizer.zero_grad(set_to_none=True)

                support_loss = torch.stack([
                    compute_loss(
                        model.forward_with_rule(pair["input"], rule),
                        pair["output"],
                    )[0]
                    for pair in support_pairs
                ]).mean()

                if not torch.isfinite(support_loss):
                    raise FloatingPointError(
                        "Support adaptation loss became non-finite."
                    )

                current_loss = support_loss.item()

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_rule = rule.detach().clone()

                if step < steps:
                    support_loss.backward()
                    optimizer.step()

        return best_rule

    finally:
        for parameter, flag in zip(parameters, original_flags):
            parameter.requires_grad_(flag)

        model.train(was_training)


# Allows the existing evaluation function to evaluate adapted predictions.
class AdaptedPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, support_pairs, query_input):
        rule = adapt_rule(self.model, support_pairs)
        return self.model.forward_with_rule(query_input, rule)


def report_families(model, episodes, adapt=False):
    if adapt:
        predictor = AdaptedPredictor(model)
        predictor.train(model.training)
    else:
        predictor = model

    results = {}

    for family in FAMILIES:
        selected = [
            episode
            for episode in episodes
            if episode["family"] == family
        ]

        report = measure(predictor, selected)

        label = (
            f"VALIDATION {family}, "
            f"{'with adaptation' if adapt else 'without adaptation'}"
        )
        show(label, report)
        results[family] = compact(report)

    return results


def synchronized_time():
    if torch.device(DEVICE).type == "mps":
        torch.mps.synchronize()

    return time.perf_counter()


# Same initialization seed, episode sequence and update count.
# Only the number of encoder passes differs.
def train_encoder_variant(rule_cycles, validation):
    model = new_model(rule_cycles)
    optimizer = make_optimizer(model)

    training_rng = random.Random(456)
    forbidden = set()

    for episode in validation:
        forbidden.update(episode_inputs(episode))

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(
        f"\nEncoder passes: {rule_cycles} | "
        f"trainable parameters: {parameter_count}",
        flush=True,
    )

    show("VALIDATION before training", measure(model, validation))
    start = synchronized_time()

    for step in range(1, STEPS + 1):
        episode = draw_episode(training_rng, forbidden=forbidden)

        training_step(
            model,
            optimizer,
            episode["support"],
            episode["query"],
        )

        if step % CHECK_EVERY == 0 or step == STEPS:
            show(
                f"VALIDATION after {step} updates",
                measure(model, validation),
            )

            save_checkpoint(
                f"encoder_cycles_{rule_cycles}.pt",
                model,
                optimizer,
                validation,
                split="validation",
                step=step,
            )

    elapsed = synchronized_time() - start

    baseline = report_families(model, validation, adapt=False)

    # Verify that adapting rule vectors leaves model weights unchanged.
    weights_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    adapted = report_families(model, validation, adapt=True)

    weights_unchanged = all(
        torch.equal(weights_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    assert weights_unchanged, "Rule adaptation changed model weights."

    return {
        "encoder_passes": rule_cycles,
        "trainable_parameters": parameter_count,
        "training_updates": STEPS,
        "training_and_periodic_validation_seconds": elapsed,
        "validation_scope": (
            "New grids from the represented synthetic rule families."
        ),
        "without_rule_adaptation": baseline,
        "with_rule_adaptation": adapted,
        "adaptation_steps": ADAPTATION_STEPS,
        "model_weights_unchanged_by_adaptation": weights_unchanged,
    }


# Preserve the existing ARC training interface.
def run_training(steps, device=DEVICE, learning_rate=LEARNING_RATE):
    if str(device) != DEVICE:
        raise ValueError("Set DEVICE at the top of this file to match device.")

    index, train_ids, val_ids, _ = checked_task_split()
    rng = random.Random(456)

    model = new_model(rule_cycles=3)
    optimizer = make_optimizer(model, learning_rate)

    for step in range(1, steps + 1):
        _, support, query = sample_episode(index, train_ids, rng)
        loss = training_step(model, optimizer, support, query)

        if step == 1 or step % CHECK_EVERY == 0:
            print(f"ARC step {step} | loss {loss:.4f}", flush=True)

    return model, index, val_ids


def evaluate_saved():
    saved = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = new_model(saved["rule_cycles"])
    model.load_state_dict(saved["model_state_dict"])

    episodes = saved["evaluation_episodes"]
    print(f"Loaded: {MODEL_FILE}")
    print(f"Evaluation split: {saved['evaluation_split']}")

    print_report(measure(model, episodes))


def main():
    RUNS.mkdir(parents=True, exist_ok=True)

    if TASK == "evaluate":
        evaluate_saved()
        return

    if TASK == "arc":
        model, index, val_ids = run_training(STEPS)
        report = evaluate(model, index, val_ids, episodes=100)
        print_report(report)

        path = RUNS / "arc_training.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "rule_cycles": 3,
                "validation_task_ids": val_ids,
                "steps": STEPS,
                "report": compact(report),
            },
            path,
        )

        print(f"Saved: {path}")
        return

    if TASK not in ("all", "fit", "compare", "split"):
        raise ValueError(f"Unknown TASK: {TASK}")

    results = {
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "synthetic_families": {
            "recolour": "Different source and target colours.",
            "rotate": "One, two or three quarter-turns.",
            "double_canvas": "Canvas expansion by factor two or three.",
        },
        "comparison_scope": (
            "One seed; equal update counts, not equal compute time."
        ),
    }

    _, _, _, split_report = checked_task_split()
    results["task_split"] = split_report
    write_report(results)

    if TASK == "split":
        return

    if TASK in ("all", "fit"):
        results["small_fitting_checks"] = {}

        for family in ("copy", "recolour"):
            print(f"\nSmall fitting check: {family}", flush=True)
            results["small_fitting_checks"][family] = fit_tiny(family)
            write_report(results)

        passed = all(
            result["passed"]
            for result in results["small_fitting_checks"].values()
        )

        if not passed:
            print(
                "\nA small fitting check did not reach the stated threshold. "
                "Results are saved. The larger comparison has not been run.",
                flush=True,
            )
            return

    if TASK in ("all", "compare"):
        validation = make_validation()
        results["encoder_comparison"] = {}

        for cycles in (1, 3):
            results["encoder_comparison"][str(cycles)] = (
                train_encoder_variant(cycles, validation)
            )
            write_report(results)

    print("\nRequested run finished. Read the saved report for the results.")


if __name__ == "__main__":
    main()