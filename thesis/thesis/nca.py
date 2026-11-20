import torch
from torch import nn
import torch.nn.functional as F

#Uses the vector from encoder.py and neighbouring cells to update the query grid 16 times 
def encode_query(grid, device):
    ids = torch.tensor(grid, dtype=torch.long, device=device)
    height, width = ids.shape

    one_hot = (
        F.one_hot(ids, num_classes=10)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )

    padding = (0, 30 - width, 0, 39 - height)
    original = F.pad(one_hot, padding)

    # Input cells have valid=1; padding cells have valid=0.
    valid = F.pad(
        torch.ones(1, 1, height, width, device=device),
        padding,
    )

    sizes = torch.tensor(
        [[height / 39, width / 30]],
        dtype=torch.float32,
        device=device,
    )

    return original, valid, sizes


class RuleNCA(nn.Module):
    def __init__(self):
        super().__init__()

        self.perception = nn.Conv2d(
            in_channels=32,
            out_channels=9 * 32,
            kernel_size=3,
            padding=1,
            groups=32,
            bias=False,
        )

        # Copy each neighbour into a separate feature.
        selectors = (
            torch.eye(9)
            .reshape(9, 1, 3, 3)
            .repeat(32, 1, 1, 1)
        )

        with torch.no_grad():
            self.perception.weight.copy_(selectors)

        self.perception.weight.requires_grad_(False)

        self.update = nn.Sequential(
            nn.Conv2d(
                9 * 32 + 10 + 1 + 64,
                128,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1),
        )

        nn.init.normal_(self.update[-1].weight, std=0.001)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, original, valid, rule):
        batch, _, height, width = original.shape

        # 10 colour channels and 22 memory channels.
        memory = original.new_zeros(batch, 22, height, width)
        state = torch.cat((original, memory), dim=1)

        # Give every cell the same rule vector.
        rule_map = rule[:, :, None, None].expand(
            -1, -1, height, width
        )

        for _ in range(16):
            features = torch.cat(
                (
                    self.perception(state),
                    original,
                    valid,
                    rule_map,
                ),
                dim=1,
            )

            change = self.update(features)
            state = state + change

        return state