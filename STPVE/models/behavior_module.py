from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 8):
        super().__init__()
        reduced_channels = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 1, kernel_size: int = 7):
        super().__init__()
        self.categorical_attention = CategoricalAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor):
        categorical_attention = self.categorical_attention(x)
        x = x * categorical_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x, categorical_attention, spatial_attention


class BehaviorCNN(nn.Module):
    """CNN encoder for viewer engagement behavior feature maps.

    The project usually stores behavior inputs as (batch, channels, history_length,
    feature_width). If a tensor arrives as (batch, history_length, channels,
    feature_width), the module detects and converts it. When flatten_multifm=True,
    MultiFM in R^{C x H x W} is converted to FM in R^{H x (C x W)}.
    """

    def __init__(
            self,
            input_size: list[int],
            behavior_kernel_size: int,
            module_out_dim: int = 256,
    ):
        super().__init__()
        channels, history_length, feature_width = input_size
        self.channels = channels
        self.history_length = history_length
        self.feature_width = feature_width
        effective_channels = channels
        effective_width = feature_width
        self.bn1d = nn.BatchNorm1d(effective_channels * effective_width, affine=False)
        attention_block = CBAMBlock
        self.attention1 = attention_block(effective_channels)
        self.conv1 = nn.Conv2d(effective_channels, 64, kernel_size=(behavior_kernel_size, behavior_kernel_size),
                               padding=(behavior_kernel_size - 1) // 2)
        self.attention2 = attention_block(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(behavior_kernel_size, behavior_kernel_size),
                               padding=(behavior_kernel_size - 1) // 2)
        self.attention3 = attention_block(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(behavior_kernel_size, behavior_kernel_size),
                               padding=(behavior_kernel_size - 1) // 2)
        self.attention4 = attention_block(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(behavior_kernel_size, behavior_kernel_size),
                               padding=(behavior_kernel_size - 1) // 2)
        self.fc = nn.Linear(32 * history_length * effective_width, module_out_dim)

    def forward(self, x: torch.Tensor):
        batch_size, channels, history_length, feature_width = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * history_length, channels * feature_width)
        x = self.bn1d(x)
        x = x.reshape(batch_size, history_length, channels, feature_width).permute(0, 2, 1, 3)
        categorical_attention, spatial_attention = None, None
        for i in range(1, 5):  # i = 1,2,3,4
            att = getattr(self, f'attention{i}')
            conv = getattr(self, f'conv{i}')
            x, ca, sa = att(x)
            if i == 1:
                categorical_attention, spatial_attention = ca, sa
            x = F.relu(conv(x))
        x = x.reshape(batch_size, -1)
        return self.fc(x), categorical_attention, spatial_attention


class VEBehaviorExtractionModule(nn.Module):
    """Extract FV^Behavior from viewer engagement behavior feature maps."""

    def __init__(self, input_size: Sequence[int], behavior_kernel_size: int, module_out_dim: int = 256):
        super().__init__()
        self.encoder = BehaviorCNN(
            list(input_size),
            behavior_kernel_size=behavior_kernel_size,
            module_out_dim=module_out_dim,
        )

    def forward(self, behavior_inputs: torch.Tensor):
        behavior_features, categorical_attention, spatial_attention = self.encoder(behavior_inputs)
        return behavior_features, categorical_attention, spatial_attention
