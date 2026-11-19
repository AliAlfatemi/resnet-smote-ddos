"""Residual model extracted from the original DDoS research script."""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.block1 = ResidualBlock(n_features, 64)
        self.block2 = ResidualBlock(64, 32, 2)
        self.linear = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assuming input is of shape [batch_size, features], we reshape it to [batch_size, features, 1] for Conv1D
        x = x.unsqueeze(2)
        out = self.block1(x)
        out = self.block2(out)
        # Global average pooling
        out = F.avg_pool1d(out, out.shape[2])
        out = out.squeeze(2)
        out = self.sigmoid(self.linear(out))
        return out
