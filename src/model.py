# src/model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyCNN(nn.Module):

    def __init__(self, vocab_size: int, channels: int = 64, dropout: float = 0.1):
        super().__init__()

        #channels controls model capacity. 64 is a good baseline for CPU/GPU.

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        #dropout helps reduce overfitting.
        self.dropout = nn.Dropout(dropout)

        # --- Policy head ---
        # need a fixed size vector before the final classifier.
        # flatten (channels*8*8) -> huge parameter count. or
        # global average pooling -> (channels,)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        #final linear classifier to vocab_size move classes
        self.fc = nn.Linear(channels, vocab_size)

        #initialize weights a bit more sensibly than default
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x expected shape: (B, 12, 8, 8)
        
        #Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        #Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        #Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        #Dropout
        x = self.dropout(x)

        #Global Average Pool -> (B, C, 1, 1)
        x = self.global_pool(x)

        #Flatten -> (B, C)
        x = x.view(x.size(0), -1)

        #Final logits -> (B, vocab_size)
        logits = self.fc(x)
        return logits