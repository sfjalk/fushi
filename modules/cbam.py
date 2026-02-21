"""
CBAM: Convolutional Block Attention Module
Paper: https://arxiv.org/abs/1807.06521

Usage (plug into any YOLOv8 backbone layer):
    from modules.cbam import CBAM
    cbam = CBAM(channels=256)
    out = cbam(feature_map)

To register with ultralytics so it can be used inside a .yaml config:
    from ultralytics.nn.tasks import attempt_load_weights
    import ultralytics.nn.modules as M
    from modules.cbam import CBAM
    M.CBAM = CBAM  # monkey-patch before model build
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention using average- and max-pooled channel features."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        scale = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(scale)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Args:
        channels:   Number of input/output channels.
        reduction:  Channel reduction ratio for the channel-attention MLP.
        kernel_size: Kernel size for the spatial-attention convolution (3 or 7).
    """

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)   # channel attention
        x = x * self.sa(x)   # spatial attention
        return x
