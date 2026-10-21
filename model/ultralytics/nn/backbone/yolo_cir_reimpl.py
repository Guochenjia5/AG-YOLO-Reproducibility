"""Independent, paper-guided YOLO-CIR reimplementation components.

This module is not author code.  It implements the architectural elements that
are stated explicitly in Zhou et al. (Infrared Physics & Technology, 2023): a
ConvNeXt feature extractor, coordinate attention in ConvNeXt blocks, spatial
attention for shallow neck features, channel attention for deep neck features,
and split attention for neck feature refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

__all__ = (
    "yolo_cir_reimpl_m",
    "CIRSpatialAttention",
    "CIRChannelAttention",
    "CIRSplitAttention",
)


class LayerNorm2d(nn.Module):
    """Layer normalization supporting NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).square().mean(1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class CIRCoordinateAttention(nn.Module):
    """Coordinate attention used inside each ConvNeXt block."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled_h = F.adaptive_avg_pool2d(x, (h, 1))
        pooled_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        fused = torch.cat((pooled_h, pooled_w), dim=2)
        fused = self.act(self.bn(self.reduce(fused)))
        attn_h, attn_w = torch.split(fused, (h, w), dim=2)
        attn_w = attn_w.transpose(2, 3)
        return x * self.height(attn_h).sigmoid() * self.width(attn_w).sigmoid()


class CIRConvNeXtBlock(nn.Module):
    """ConvNeXt-v1 block followed by coordinate attention."""

    def __init__(self, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.coord_attn = CIRCoordinateAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.project(self.act(self.expand(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return self.coord_attn(x)


class CIRBackbone(nn.Module):
    """Four-stage ConvNeXt backbone exposing P2--P5 feature maps."""

    def __init__(
        self,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (80, 160, 320, 640),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.channel = list(dims)
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0]))]
        )
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[index]),
                    nn.Conv2d(dims[index], dims[index + 1], 2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for channels, count in zip(dims, depths):
            self.stages.append(
                nn.Sequential(*(CIRConvNeXtBlock(channels, rates[offset + j]) for j in range(count)))
            )
            offset += count
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            outputs.append(x)
        return outputs


def yolo_cir_reimpl_m() -> CIRBackbone:
    """Medium YOLO-CIR proxy backbone, sized near the reported 35.9 M detector."""

    return CIRBackbone()


class CIRSpatialAttention(nn.Module):
    """Spatial attention for the shallow, high-resolution neck feature."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        del channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * self.conv(context).sigmoid()


class CIRChannelAttention(nn.Module):
    """Channel attention for deep, semantic neck features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CIRSplitAttention(nn.Module):
    """Two-radix split-attention refinement used after neck fusion blocks."""

    def __init__(self, channels: int, radix: int = 2, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.radix = radix
        hidden = max(channels * radix // reduction, 32)
        self.branches = nn.Sequential(
            nn.Conv2d(channels, channels * radix, 3, padding=1, groups=radix, bias=False),
            nn.BatchNorm2d(channels * radix),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, channels * radix, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.branches(x).view(x.shape[0], self.radix, self.channels, *x.shape[-2:])
        pooled = split.sum(dim=1).mean(dim=(2, 3), keepdim=True)
        logits = self.fc2(F.relu(self.bn1(self.fc1(pooled)), inplace=True))
        weights = logits.view(x.shape[0], self.radix, self.channels, 1, 1).softmax(dim=1)
        return (split * weights).sum(dim=1)
