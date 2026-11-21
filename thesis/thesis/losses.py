import torch
import torch.nn.functional as F


def compute_loss(prediction, target_grid):
    device = prediction["colour_logits"].device

    target = torch.tensor(
        target_grid,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    _, height, width = target.shape

    colour_loss = F.cross_entropy(
        prediction["colour_logits"][:, :, :height, :width],
        target,
    )

    height_loss = F.cross_entropy(
        prediction["height_logits"],
        torch.tensor([height - 1], device=device),
    )

    width_loss = F.cross_entropy(
        prediction["width_logits"],
        torch.tensor([width - 1], device=device),
    )

    parts = {
        "colour": colour_loss,
        "height": height_loss,
        "width": width_loss,
    }

    return colour_loss + height_loss + width_loss, parts