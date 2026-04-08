from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyCNN(nn.Module):
    def __init__(self, vocab_size: int, channels: int = 64, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = channels + 6

        # policy head
        self.policy_head = nn.Linear(feature_dim, vocab_size)

        # value head
        self.value_head = nn.Linear(feature_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, extras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        features = torch.cat([x, extras], dim=1)

        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).squeeze(1)

        return policy_logits, value