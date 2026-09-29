"""
Neural Network Layers and Blocks for ceVAE+.
Implements:
1. DownsampleBlock: Conv2d(4x4, stride 2, pad 1) + InstanceNorm2d + LeakyReLU(0.2)
2. UpsampleBlock: Bilinear 2x upsampling + Conv2d(3x3, stride 1, pad 1) + InstanceNorm2d + LeakyReLU(0.2)
   Eliminates checkerboard artifacts inherent to ConvTranspose2d.
"""

import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    """
    Convolutional downsampling stage:
    Conv2d(kernel_size=4, stride=2, padding=1) -> InstanceNorm2d -> LeakyReLU(0.2)
    """

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    Checkerboard-free upsampling stage:
    Bilinear 2x Interpolation -> Conv2d(kernel_size=3, stride=1, padding=1) -> InstanceNorm2d -> LeakyReLU(0.2)
    """

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)
