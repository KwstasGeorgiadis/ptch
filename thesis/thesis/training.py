import random

import torch
from torch import nn

from .data import index_task_files, sample_episode, split_tasks
from .losses import compute_loss
from .model import Architecture


def training_step(model, optimizer, support, query):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(support, query["input"])
    loss, _ = compute_loss(prediction, query["output"])

    loss.backward()

    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    return loss.item()


def run_training(steps, device="mps"):
    torch.manual_seed(0)
    rng = random.Random(1)

    index = index_task_files()
    train_ids, val_ids = split_tasks(index, seed=0)

    model = Architecture().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for step in range(1, steps + 1):
        _, support, query = sample_episode(
            index, train_ids, rng
        )

        loss = training_step(
            model, optimizer, support, query
        )

        if step == 1 or step % 100 == 0:
            print(
                f"Step {step} | loss {loss:.4f}",
                flush=True,
            )

    return model, index, val_ids