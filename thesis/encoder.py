import torch
from torch import nn

#The whole code reads solved examples applies two transformers blocks repeatedly and produces a 64-num rule vector 
class GridEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.color_embedding = nn.Embedding(10, 64)
        self.row_embedding = nn.Embedding(39, 64)
        self.column_embedding = nn.Embedding(30, 64)
        self.role_embedding = nn.Embedding(2, 64)

    def forward(self, grid_ids, role_id):
        # grid_ids: [batch, height, width]
        # role_id: 0 = support input, 1 = support output
        _, height, width = grid_ids.shape

        rows = torch.arange(
            height, device=grid_ids.device
        ).view(1, height, 1)

        cols = torch.arange(
            width, device=grid_ids.device
        ).view(1, 1, width)

        roles = torch.full_like(grid_ids, role_id)

        tokens = (
            self.color_embedding(grid_ids)
            + self.row_embedding(rows)
            + self.column_embedding(cols)
            + self.role_embedding(roles)
        )

        return tokens.flatten(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(64)

        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,#maybe 4 
            dropout=0.0,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(64)

        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        normalized = self.norm1(x)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        x = x + attended

        return x + self.mlp(self.norm2(x))


class RuleEncoder(nn.Module):
    def __init__(self, rule_cycles=10):
        super().__init__()

        self.rule_cycles = rule_cycles
        self.grid_encoder = GridEncoder()

        self.stack = nn.Sequential(
            TransformerBlock(),
            TransformerBlock(),
        )

        self.rule_projection = nn.Linear(64, 64)

    def forward(self, support_pairs):
        device = self.rule_projection.weight.device
        pair_vectors = []

        for pair in support_pairs:
            input_ids = torch.tensor(
                pair["input"],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            output_ids = torch.tensor(
                pair["output"],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            embeddings = torch.cat(
                (
                    self.grid_encoder(input_ids, 0),
                    self.grid_encoder(output_ids, 1),
                ),
                dim=1,
            )

            state = torch.zeros_like(embeddings)

            for _ in range(self.rule_cycles):
                state = self.stack(state + embeddings)

            pair_vectors.append(state.mean(dim=1))

        pooled = torch.stack(pair_vectors, dim=0).mean(dim=0)

        return self.rule_projection(pooled)