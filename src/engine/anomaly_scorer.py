"""
Diagnostic Anomaly Scorer for ceVAE+.
Implements the Zimmerer et al. (2019) dual-scoring mechanism:
A_pixel = GaussianBlur_{sigma=1.5}( |x - \\hat{x}| \\odot |\\partial L_KL / \\partial x| )

Extracts input-level autograd saliency of KL divergence to suppress edge false positives
and enhance pathological signal.
"""

from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.cevae_loss import CompositeCeVAELoss


def gaussian_blur_2d(
    img: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Applies 2D Gaussian blur filter using depthwise convolution.
    Args:
        img: Tensor of shape (B, C, H, W)
        kernel_size: Odd integer kernel size (default 7)
        sigma: Standard deviation of Gaussian kernel (default 1.5)
    Returns:
        Smoothed tensor of shape (B, C, H, W)
    """
    channels = img.size(1)
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g_1d = g_1d / g_1d.sum()
    g_2d = torch.outer(g_1d, g_1d).unsqueeze(0).unsqueeze(0)
    kernel = g_2d.repeat(channels, 1, 1, 1)

    pad = kernel_size // 2
    return F.conv2d(img, kernel, padding=pad, groups=channels)


class DiagnosticAnomalyScorer:
    """
    Computes diagnostic anomaly maps and saliency signals for input MRI slices.
    """

    def __init__(
        self,
        model: nn.Module,
        gaussian_sigma: float = 1.5,
        gaussian_kernel_size: int = 7,
    ):
        self.model = model
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size

    def score_slice(
        self,
        x: torch.Tensor,
        normalize_scores: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes dual-scoring pipeline on input slice x.
        Args:
            x: Tensor of shape (B, 1, H, W)
        Returns:
            Dict containing:
                - "reconstruction": \\hat{x} (B, 1, H, W)
                - "residual": |x - \\hat{x}| (B, 1, H, W)
                - "kl_saliency": |\\partial L_KL / \\partial x| (B, 1, H, W)
                - "raw_composite": residual * kl_saliency (B, 1, H, W)
                - "anomaly_map": Gaussian smoothed composite map (B, 1, H, W)
                - "slice_score": Image-level anomaly score (mean of top 5% pixels)
        """
        self.model.eval()

        # Ensure requires_grad is True for input slice
        x_in = x.clone().detach().requires_grad_(True)

        # Forward pass
        z, mu, logvar = self.model.encode(x_in)
        recon = self.model.decode(z)

        # 1. Spatial reconstruction residual
        residual = torch.abs(x_in - recon)

        # 2. Input-level KL gradient saliency
        # Analytical KL divergence per sample
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        kl_sum = kl_per_sample.sum()

        # Compute gradient d(L_KL) / dx
        kl_grad = torch.autograd.grad(
            outputs=kl_sum,
            inputs=x_in,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        kl_saliency = torch.abs(kl_grad)

        # Optional normalization of saliency to [0, 1] per slice
        if normalize_scores:
            b, c, h, w = kl_saliency.shape
            s_flat = kl_saliency.view(b, -1)
            s_min = s_flat.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
            s_max = s_flat.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
            kl_saliency_norm = (kl_saliency - s_min) / (s_max - s_min + 1e-8)
        else:
            kl_saliency_norm = kl_saliency

        # 3. Dual-scoring composite: Residual * KL Saliency
        raw_composite = residual * kl_saliency_norm

        # 4. Gaussian Blur Smoothing
        anomaly_map = gaussian_blur_2d(
            raw_composite,
            kernel_size=self.gaussian_kernel_size,
            sigma=self.gaussian_sigma,
        )

        # 5. Image-level score: Top-K average (focuses on focal anomalies rather than diffuse noise)
        b = x.shape[0]
        a_flat = anomaly_map.view(b, -1)
        k = max(1, int(0.05 * a_flat.shape[1]))  # top 5%
        topk_vals = torch.topk(a_flat, k=k, dim=-1)[0]
        slice_score = topk_vals.mean(dim=-1)

        return {
            "reconstruction": recon.detach(),
            "residual": residual.detach(),
            "kl_saliency": kl_saliency.detach(),
            "raw_composite": raw_composite.detach(),
            "anomaly_map": anomaly_map.detach(),
            "slice_score": slice_score.detach(),
        }
