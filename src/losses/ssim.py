"""
Differentiable Structural Similarity Index (SSIM) Loss in Pure PyTorch.
Computes local luminance, contrast, and structural comparison using a Gaussian window.
No external library dependencies required.
"""

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
    window = g_2d.repeat(channels, 1, 1, 1)
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss: L_SSIM = 1 - SSIM(x, y).
    Range: [0.0, 1.0] where 0.0 denotes identical structural fidelity.
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

        # Register Gaussian window as a buffer so it moves across devices automatically
        window = create_gaussian_window_2d(window_size, sigma, in_channels)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: Tensors of shape (B, C, H, W) normalized to [0.0, 1.0]
        Returns:
            Scalar tensor representing (1 - mean_SSIM)
        """
        channels = x.size(1)
        if channels != self.in_channels:
            # Recreate window if channel count differs
            window = create_gaussian_window_2d(self.window_size, self.sigma, channels).to(x.device)
        else:
            window = self.window

        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=channels)
        mu_y = F.conv2d(y, window, padding=pad, groups=channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

        # Numerical stabilization clamp
        sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

        numerator = (2.0 * mu_xy + self.c1) * (2.0 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2)

        ssim_map = numerator / (denominator + 1e-8)
        mean_ssim = ssim_map.mean()

        return 1.0 - mean_ssim
