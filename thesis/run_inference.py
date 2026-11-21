import json
import random
from pathlib import Path

import torch

from .data import index_task_files, sample_episode
from .evaluation import decode_prediction
from .model import Architecture

MODEL_FILE = Path("runs/arc_training.pt")
DEVICE = "mps"


def main():
    saved = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = Architecture().to(DEVICE)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    if "support" in saved:
        print("Example from the fixed training set.")

        support = saved["support"]
        query = saved["queries"][0]

    else:
        index = index_task_files()

        task_id, support, query = sample_episode(
            index,
            saved["validation_task_ids"],
            random.Random(123),
        )

        print(f"Validation task: {task_id}")

    with torch.no_grad():
        prediction = model(support, query["input"])
        output = decode_prediction(prediction)

    print(json.dumps({
        "input": query["input"],
        "prediction": output,
        "target": query["output"],
    }, indent=2))


if __name__ == "__main__":
    main()