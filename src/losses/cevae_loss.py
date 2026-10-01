"""
Loss Engine for ceVAE / Spatial VAE.
Implements:
1. Stable Pixel Reconstruction Loss:
   L_recon = mse_weight * MSE + l1_weight * L1 (+ optional auxiliary SSIM/Edge/BCE terms)
2. Spatial Analytical KL Divergence:
   D_KL(q(z|x) || p(z)) normalized across spatial latent dimensions.
3. Dual-Pass / Single-Pass Objective:
   L_total = clean_weight * (L_recon_clean + beta * L_KL) + inpaint_weight * L_recon_inpaint
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from src.losses.ssim import SSIMLoss


class CompositeCeVAELoss(nn.Module):
    """
    Robust, simplified loss engine for medical VAE reconstruction and anomaly detection.
    Default formulation uses stable MSE + L1 reconstruction loss with balanced KL divergence,
    avoiding BCE gradient saturation or SSIM boundary artifacts.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        bce_weight: float = 0.0,
        ssim_weight: float = 0.0,
        edge_weight: float = 0.0,
        beta_kl: float = 0.001,
        clean_weight: float = 1.0,
        inpaint_weight: float = 0.5,
        window_size: int = 11,
        in_channels: int = 1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.bce_weight = bce_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.beta_kl = beta_kl
        self.clean_weight = clean_weight
        self.inpaint_weight = inpaint_weight

        # Primary well-behaved reconstruction losses
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")

        # SSIM calculation (always tracked for diagnostics, weighted only if ssim_weight > 0)
        self.ssim_loss = SSIMLoss(window_size=window_size, in_channels=in_channels)

        # Optional BCE loss (only if explicitly enabled)
        if self.bce_weight > 0:
            self.bce_loss = nn.BCELoss(reduction="mean")
        else:
            self.bce_loss = None

    @staticmethod
    def edge_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        High-frequency gradient difference to penalize boundary blurriness.
        """
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        dx_pred = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dy_pred = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        return torch.mean(torch.abs(dx_target - dx_pred)) + torch.mean(torch.abs(dy_target - dy_pred))

    def reconstruction_loss(
        self, target: torch.Tensor, prediction: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Base stable reconstruction: MSE + L1
        mse = self.mse_loss(prediction, target)
        l1 = self.l1_loss(prediction, target)
        loss_recon = self.mse_weight * mse + self.l1_weight * l1

        # 2. Optional auxiliary terms (active only if positive weight specified)
        if self.bce_weight > 0 and self.bce_loss is not None:
            pred_clamped = torch.clamp(prediction, 1e-6, 1.0 - 1e-6)
            loss_recon = loss_recon + self.bce_weight * self.bce_loss(pred_clamped, target)

        if self.edge_weight > 0:
            loss_recon = loss_recon + self.edge_weight * self.edge_loss(target, prediction)

        # 3. Track SSIM term
        ssim_term = self.ssim_loss(prediction, target)
        if self.ssim_weight > 0:
            loss_recon = loss_recon + self.ssim_weight * ssim_term

        return loss_recon, l1, ssim_term

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Analytical KL divergence for standard normal prior N(0, I).
        Supports both 1D (B, D) and spatial (B, C, H, W) latent representations.
        Normalized per latent element for stability across spatial grids.
        """
        if mu.dim() > 2:
            kl_per_sample = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=(1, 2, 3))
        else:
            kl_per_sample = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        return torch.mean(kl_per_sample)

    def forward(
        self,
        target_clean: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        current_beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        beta = current_beta if current_beta is not None else self.beta_kl

        # Clean reconstruction loss
        recon_clean = model_outputs["recon_clean"]
        l_recon_clean, l1_clean, ssim_clean = self.reconstruction_loss(target_clean, recon_clean)

        # Spatial KL divergence
        mu = model_outputs["mu"]
        logvar = model_outputs["logvar"]
        l_kl = self.kl_divergence(mu, logvar)

        # Inpainting reconstruction loss (if inpainting pass was computed)
        if "recon_inpaint" in model_outputs and model_outputs["recon_inpaint"] is not None:
            recon_inpaint = model_outputs["recon_inpaint"]
            l_recon_inpaint, _, _ = self.reconstruction_loss(target_clean, recon_inpaint)

            l_total = (
                self.clean_weight * (l_recon_clean + beta * l_kl)
                + self.inpaint_weight * l_recon_inpaint
            )
        else:
            l_recon_inpaint = torch.tensor(0.0, device=target_clean.device)
            l_total = self.clean_weight * (l_recon_clean + beta * l_kl)

        return {
            "loss_total": l_total,
            "loss_recon_clean": l_recon_clean,
            "loss_recon_inpaint": l_recon_inpaint,
            "loss_l1": l1_clean,
            "loss_ssim": ssim_clean,
            "loss_kl": l_kl,
            "beta": torch.tensor(beta, device=target_clean.device),
        }


class SimpleVAELoss(nn.Module):
    """
    Minimalist VAE Loss: MSE reconstruction + beta * KL divergence.
    Fast, rock-solid, and guaranteed to visibly reconstruct normative images.
    """

    def __init__(self, beta_kl: float = 0.001):
        super().__init__()
        self.beta_kl = beta_kl
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")

    def forward(
        self,
        target_clean: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        current_beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        beta = current_beta if current_beta is not None else self.beta_kl
        recon_clean = model_outputs["recon_clean"]
        mse = self.mse_loss(recon_clean, target_clean)
        l1 = self.l1_loss(recon_clean, target_clean)

        mu = model_outputs["mu"]
        logvar = model_outputs["logvar"]
        if mu.dim() > 2:
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar))
        else:
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1).mean()

        total = mse + beta * kl

        return {
            "loss_total": total,
            "loss_recon_clean": mse,
            "loss_recon_inpaint": torch.tensor(0.0, device=target_clean.device),
            "loss_l1": l1,
            "loss_ssim": torch.tensor(0.0, device=target_clean.device),
            "loss_kl": kl,
            "beta": torch.tensor(beta, device=target_clean.device),
        }
