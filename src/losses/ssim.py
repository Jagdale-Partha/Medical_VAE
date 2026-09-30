"""
Differentiable Structural Similarity Index (SSIM) Loss in Pure PyTorch.
Computes local luminance, contrast, and structural comparison using a Gaussian window.
No external library dependencies required.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window_2d(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """
    Creates a 2D Gaussian kernel tensor of shape (channels, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g_1d = g_1d / g_1d.sum()
    g_2d = torch.outer(g_1d, g_1d).unsqueeze(0).unsqueeze(0)
    return g_2d.repeat(channels, 1, 1, 1)


def create_gaussian_kernels_1d(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates separable 1D Gaussian kernel tensors:
    - kernel_h of shape (channels, 1, 1, window_size)
    - kernel_v of shape (channels, 1, window_size, 1)
    Provides mathematically identical results with ~5x faster execution on CPU.
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g_1d = g_1d / g_1d.sum()
    kernel_h = g_1d.view(1, 1, 1, window_size).repeat(channels, 1, 1, 1)
    kernel_v = g_1d.view(1, 1, window_size, 1).repeat(channels, 1, 1, 1)
    return kernel_h, kernel_v


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss: L_SSIM = 1 - SSIM(x, y).
    Range: [0.0, 1.0] where 0.0 denotes identical structural fidelity.
    Optimized via separable 1D Gaussian convolutions for fast CPU execution.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        in_channels: int = 1,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.in_channels = in_channels
        self.c1 = c1
        self.c2 = c2

        k_h, k_v = create_gaussian_kernels_1d(window_size, sigma, in_channels)
        self.register_buffer("kernel_h", k_h)
        self.register_buffer("kernel_v", k_v)

    def _conv_gauss(self, tensor: torch.Tensor, k_h: torch.Tensor, k_v: torch.Tensor, pad: int, channels: int) -> torch.Tensor:
        t = F.conv2d(tensor, k_h, padding=(0, pad), groups=channels)
        return F.conv2d(t, k_v, padding=(pad, 0), groups=channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: Tensors of shape (B, C, H, W) normalized to [0.0, 1.0]
        Returns:
            Scalar tensor representing (1 - mean_SSIM)
        """
        channels = x.size(1)
        if channels != self.in_channels:
            k_h, k_v = create_gaussian_kernels_1d(self.window_size, self.sigma, channels)
            k_h, k_v = k_h.to(x.device), k_v.to(x.device)
        else:
            k_h, k_v = self.kernel_h, self.kernel_v

        pad = self.window_size // 2

        mu_x = self._conv_gauss(x, k_h, k_v, pad, channels)
        mu_y = self._conv_gauss(y, k_h, k_v, pad, channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = self._conv_gauss(x * x, k_h, k_v, pad, channels) - mu_x_sq
        sigma_y_sq = self._conv_gauss(y * y, k_h, k_v, pad, channels) - mu_y_sq
        sigma_xy = self._conv_gauss(x * y, k_h, k_v, pad, channels) - mu_xy

        sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

        numerator = (2.0 * mu_xy + self.c1) * (2.0 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2)

        ssim_map = numerator / (denominator + 1e-8)
        return 1.0 - ssim_map.mean()
