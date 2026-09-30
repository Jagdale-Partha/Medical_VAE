"""
Loss Engine for Spatial ceVAE.
Implements:
1. Sharpness-preserving Reconstruction loss:
   L_recon = 0.7 * L1 + 0.2 * (1 - SSIM) + 0.1 * L_edge
2. Spatial Analytical KL divergence: D_KL(q(z|x) || p(z))
   Normalized across spatial latent dimensions to maintain optimal balance with reconstruction.
3. Composite context-inpainting objective:
   L_total = 0.5 * (L_recon(x, x_clean) + beta * L_KL) + 0.5 * L_recon(x, x_inpaint)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.losses.ssim import SSIMLoss


class CompositeCeVAELoss(nn.Module):
    """
    Sharpness-preserving loss engine combining dual-pass reconstruction,
    high-frequency gradient preservation, and spatial latent regularization.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        l1_weight: float = 0.5,
        ssim_weight: float = 0.2,
        edge_weight: float = 0.1,
        beta_kl: float = 0.002,
        clean_weight: float = 0.5,
        inpaint_weight: float = 0.5,
        window_size: int = 11,
        in_channels: int = 1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.beta_kl = beta_kl
        self.clean_weight = clean_weight
        self.inpaint_weight = inpaint_weight

        self.bce_loss = nn.BCELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=window_size, in_channels=in_channels)

    @staticmethod
    def edge_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        High-frequency gradient difference to penalize boundary blurriness
        and enforce razor-sharp sulcal and ventricular contours.
        """
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        dx_pred = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dy_pred = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        return torch.mean(torch.abs(dx_target - dx_pred)) + torch.mean(torch.abs(dy_target - dy_pred))

    def reconstruction_loss(
        self, target: torch.Tensor, prediction: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_clamped = torch.clamp(prediction, 1e-6, 1.0 - 1e-6)
        bce_term = self.bce_loss(pred_clamped, target)
        l1 = self.l1_loss(prediction, target)
        ssim_term = self.ssim_loss(prediction, target)
        edge_term = self.edge_loss(target, prediction)
        loss_recon = (
            self.bce_weight * bce_term
            + self.l1_weight * l1
            + self.ssim_weight * ssim_term
            + self.edge_weight * edge_term
        )
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
        current_beta: float = None,
    ) -> Dict[str, torch.Tensor]:
        beta = current_beta if current_beta is not None else self.beta_kl

        # Clean reconstruction loss
        recon_clean = model_outputs["recon_clean"]
        l_recon_clean, l1_clean, ssim_clean = self.reconstruction_loss(target_clean, recon_clean)

        # Spatial KL divergence
        mu = model_outputs["mu"]
        logvar = model_outputs["logvar"]
        l_kl = self.kl_divergence(mu, logvar)

        # Inpainting reconstruction loss
        if "recon_inpaint" in model_outputs and model_outputs["recon_inpaint"] is not None:
            recon_inpaint = model_outputs["recon_inpaint"]
            l_recon_inpaint, l1_inpaint, ssim_inpaint = self.reconstruction_loss(target_clean, recon_inpaint)

            l_total = (
                self.clean_weight * (l_recon_clean + beta * l_kl)
                + self.inpaint_weight * l_recon_inpaint
            )
        else:
            l_recon_inpaint = torch.tensor(0.0, device=target_clean.device)
            l_total = l_recon_clean + beta * l_kl

        return {
            "loss_total": l_total,
            "loss_recon_clean": l_recon_clean,
            "loss_recon_inpaint": l_recon_inpaint,
            "loss_l1": l1_clean,
            "loss_ssim": ssim_clean,
            "loss_kl": l_kl,
            "beta": torch.tensor(beta, device=target_clean.device),
        }
