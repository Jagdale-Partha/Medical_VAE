"""
Loss Engine for ceVAE+.
Implements:
1. Reconstruction loss: L_recon = 0.8 * L1 + 0.2 * (1 - SSIM)
2. Analytical KL divergence: D_KL(q(z|x) || p(z))
3. Composite context-inpainting objective:
   L_total = 0.5 * (L_recon(x, x_clean) + beta * L_KL) + 0.5 * L_recon(x, x_inpaint)
   with beta = 0.001.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.losses.ssim import SSIMLoss


class CompositeCeVAELoss(nn.Module):
    """
    Full loss function combining dual-pass reconstruction and latent regularization.
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        beta_kl: float = 0.001,
        clean_weight: float = 0.5,
        inpaint_weight: float = 0.5,
        window_size: int = 11,
        in_channels: int = 1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.beta_kl = beta_kl
        self.clean_weight = clean_weight
        self.inpaint_weight = inpaint_weight

        self.l1_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=window_size, in_channels=in_channels)

    def reconstruction_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes L_recon = 0.8 * L1 + 0.2 * (1 - SSIM)
        """
        l1 = self.l1_loss(prediction, target)
        ssim_term = self.ssim_loss(prediction, target)
        loss_recon = self.l1_weight * l1 + self.ssim_weight * ssim_term
        return loss_recon, l1, ssim_term

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Analytical KL divergence for standard multivariate normal prior N(0, I):
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        Averaged across the batch.
        """
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        return torch.mean(kl_per_sample)

    def forward(
        self,
        target_clean: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        current_beta: float = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            target_clean: Original ground-truth slice x in [0.0, 1.0]
            model_outputs: Dict containing "recon_clean", "mu", "logvar", and optional "recon_inpaint"
            current_beta: Optional override for KL beta (e.g. for beta-annealing warmup)
        """
        beta = current_beta if current_beta is not None else self.beta_kl

        # Clean reconstruction loss
        recon_clean = model_outputs["recon_clean"]
        l_recon_clean, l1_clean, ssim_clean = self.reconstruction_loss(target_clean, recon_clean)

        # Analytical KL divergence
        mu = model_outputs["mu"]
        logvar = model_outputs["logvar"]
        l_kl = self.kl_divergence(mu, logvar)

        # Inpainting reconstruction loss (if available)
        if "recon_inpaint" in model_outputs and model_outputs["recon_inpaint"] is not None:
            recon_inpaint = model_outputs["recon_inpaint"]
            l_recon_inpaint, l1_inpaint, ssim_inpaint = self.reconstruction_loss(target_clean, recon_inpaint)

            l_total = (
                self.clean_weight * (l_recon_clean + beta * l_kl)
                + self.inpaint_weight * l_recon_inpaint
            )
        else:
            # Fallback for inference or standard VAE baseline
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
