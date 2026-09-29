"""
ceVAE+ (Enhanced Context-Encoding Variational Autoencoder) Architecture.
Bridges trade-offs in Baur et al. (2021) and Zimmerer et al. (2019):
- Dense bottleneck (16384 -> 128) with NO spatial skip connections to eliminate pathology leakage.
- InstanceNorm2d + Bilinear Upsampling + Conv to prevent checkerboard artifacts and edge blur.
- Dual-pass context inpainting training + input-level KL gradient saliency inference.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from src.models.layers import DownsampleBlock, UpsampleBlock


class Encoder(nn.Module):
    """
    4-stage convolutional downsampling encoder.
    Maps 1 x 128 x 128 -> 256 x 8 x 8 feature map.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, negative_slope: float = 0.2):
        super().__init__()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256

        self.stage1 = DownsampleBlock(in_channels, c1, negative_slope=negative_slope)  # 128 -> 64
        self.stage2 = DownsampleBlock(c1, c2, negative_slope=negative_slope)           # 64 -> 32
        self.stage3 = DownsampleBlock(c2, c3, negative_slope=negative_slope)           # 32 -> 16
        self.stage4 = DownsampleBlock(c3, c4, negative_slope=negative_slope)           # 16 -> 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class Bottleneck(nn.Module):
    """
    Dense 1D bottleneck with reparameterization trick.
    Maps 256 x 8 x 8 (16384) -> mu, logvar in R^128 -> sampled latent z in R^128.
    Strictly avoids spatial skip connections to prevent lesion leakage.
    """

    def __init__(self, flattened_dim: int = 16384, latent_dim: int = 128):
        super().__init__()
        self.flattened_dim = flattened_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = mu + eps * sigma, eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Tensor (B, 256, 8, 8)
        Returns:
            z: Sampled latent vector (B, 128)
            mu: Mean vector (B, 128)
            logvar: Log-variance vector (B, 128)
        """
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        # Numerical safeguard on logvar
        logvar = torch.clamp(logvar, min=-15.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    """
    Checkerboard-free upsampling decoder.
    Expands latent z (128) -> 256 x 8 x 8 -> 4 bilinear upsample stages -> 1 x 128 x 128.
    Final activation: Sigmoid() for intensity range [0.0, 1.0].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        flattened_dim: int = 16384,
        base_channels: int = 32,
        out_channels: int = 1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.c4 = base_channels * 8  # 256

        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256

        self.up1 = UpsampleBlock(c4, c3, negative_slope=negative_slope)  # 8 -> 16
        self.up2 = UpsampleBlock(c3, c2, negative_slope=negative_slope)  # 16 -> 32
        self.up3 = UpsampleBlock(c2, c1, negative_slope=negative_slope)  # 32 -> 64
        self.up4 = UpsampleBlock(c1, c1, negative_slope=negative_slope)  # 64 -> 128

        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(-1, self.c4, 8, 8)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        x_recon = self.final_conv(h)
        return x_recon


class EnhancedContextVAE(nn.Module):
    """
    Enhanced Context-Encoding Variational Autoencoder (ceVAE+).
    Integrates:
    - Encoder + Bottleneck + Decoder
    - Clean forward pass + Inpainting forward pass
    - Differentiable analytical KL computation
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 128,
        base_channels: int = 32,
        latent_dim: int = 128,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, negative_slope=negative_slope)
        flattened_dim = (base_channels * 8) * 8 * 8  # 256 * 8 * 8 = 16384
        self.bottleneck = Bottleneck(flattened_dim=flattened_dim, latent_dim=latent_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            flattened_dim=flattened_dim,
            base_channels=base_channels,
            out_channels=in_channels,
            negative_slope=negative_slope,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        z, mu, logvar = self.bottleneck(features)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x_clean: torch.Tensor, x_masked: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes dual forward pass:
        1. Clean pass: x_clean -> z_clean -> x_recon_clean
        2. Inpaint pass (if x_masked provided): x_masked -> z_masked -> x_recon_inpaint
        """
        # Pass 1: Clean
        z_clean, mu_clean, logvar_clean = self.encode(x_clean)
        recon_clean = self.decode(z_clean)

        outputs = {
            "recon_clean": recon_clean,
            "mu": mu_clean,
            "logvar": logvar_clean,
            "z": z_clean,
        }

        # Pass 2: Inpainted pass
        if x_masked is not None:
            z_masked, mu_masked, logvar_masked = self.encode(x_masked)
            recon_inpaint = self.decode(z_masked)
            outputs["recon_inpaint"] = recon_inpaint
            outputs["mu_masked"] = mu_masked
            outputs["logvar_masked"] = logvar_masked

        return outputs
