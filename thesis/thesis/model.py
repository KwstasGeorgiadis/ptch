import torch
from torch import nn

from .encoder import RuleEncoder
from .nca import RuleNCA, encode_query

#Connects encoder.py and nca.py and adds the layers that predict the output 
class Architecture(nn.Module):
    def __init__(self):
        super().__init__()

        self.trm = RuleEncoder()
        self.nca = RuleNCA()

        self.size_features = nn.Sequential(
            nn.Linear(32 + 64 + 2, 128),
            nn.ReLU(),
        )

        self.height_head = nn.Linear(128, 39)
        self.width_head = nn.Linear(128, 30)

    def forward_with_rule(self, query_input, rule):
        original, valid, input_sizes = encode_query(
            query_input, rule.device
        )

        state = self.nca(original, valid, rule)

        summary = torch.cat(
            (
                state.mean(dim=(2, 3)),
                rule,
                input_sizes,
            ),
            dim=1,
        )

        size_hidden = self.size_features(summary)

        return {
            "colour_logits": state[:, :10],
            "height_logits": self.height_head(size_hidden),
            "width_logits": self.width_head(size_hidden),
        }

    def forward(self, support_pairs, query_input):
        rule = self.trm(support_pairs)

        return self.forward_with_rule(query_input, rule)