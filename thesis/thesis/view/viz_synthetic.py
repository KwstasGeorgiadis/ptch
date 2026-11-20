from pathlib import Path

import torch

from ..model import Architecture
from .visualization import plot_predictions

MODEL_FILE = Path("runs/double_canvas_fresh.pt")#("runs/add_one.pt")
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

    examples = [
        (saved["support"], query)
        for query in saved["queries"]
    ]

    plot_predictions(model, examples)


if __name__ == "__main__":
    main()