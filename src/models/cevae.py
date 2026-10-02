"""
Spatial ceVAE (Spatial Context-Encoding Variational Autoencoder) Architecture.
Solves the fundamental blurriness of 1D bottleneck VAEs in Medical Brain MRI:
- Preserves 2D topological feature grids in latent space (8 x 16 x 16) instead of collapsing into a 1D vector.
- Eliminates the 2.1-million parameter dense linear bottleneck that acted as an aggressive low-pass blur filter.
- Retains 64x spatial area compression (8x downsampling) to prevent pathological lesion leakage.
- Incorporates residual refinement blocks to preserve fine sulcal, gyral, and ventricular boundaries.
- Preserves context-encoding (ceVAE) dual-pass inpainting for robust unsupervised anomaly detection.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn


class ResBlock2d(nn.Module):
    """
    Residual block with InstanceNorm2d and LeakyReLU for sharp anatomical edge preservation.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))
        return residual + out


class Encoder(nn.Module):
    """
    Hierarchical 3-stage spatial downsampling encoder.
    Maps 1 x 128 x 128 input -> 128 x 16 x 16 feature map.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, negative_slope: float = 0.2):
        super().__init__()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        # Stage 1: 128 -> 64
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c2, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        # Stage 2: 64 -> 32
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        # Stage 3: 32 -> 16 with residual refinement
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            ResBlock2d(c3, negative_slope=negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return h


class Bottleneck(nn.Module):
    """
    Spatial VAE Bottleneck.
    Maps 128 x 16 x 16 feature map -> mu, logvar of shape (B, latent_channels, 16, 16).
    Preserves 2D spatial locality and topological neighborhoods.
    """

    def __init__(self, in_channels: int = 128, latent_channels: int = 8):
        super().__init__()
        self.latent_channels = latent_channels
        self.conv_mu = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        self.conv_logvar = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.conv_mu(h)
        logvar = torch.clamp(self.conv_logvar(h), min=-15.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    """
    Hierarchical 3-stage spatial upsampling decoder with residual refinement.
    Maps latent z (B, 8, 16, 16) -> 1 x 128 x 128 sharp reconstruction.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 32,
        out_channels: int = 1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128

        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_channels, c3, kernel_size=1),
            ResBlock2d(c3, negative_slope=negative_slope),
        )
        # Stage 1: 16 -> 32
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            ResBlock2d(c3, negative_slope=negative_slope),
        )
        # Stage 2: 32 -> 64
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c2, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            ResBlock2d(c2, negative_slope=negative_slope),
        )
        # Stage 3: 64 -> 128 with edge refinement
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            ResBlock2d(c1, negative_slope=negative_slope),
            nn.InstanceNorm2d(c1, affine=True),
        )
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Initialize final conv to prevent early saturation
        nn.init.kaiming_normal_(self.final_conv[0].weight, mode="fan_out", nonlinearity="linear")
        self.final_conv[0].weight.data.mul_(0.1)
        self.final_conv[0].bias.data.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return self.final_conv(h)


class EnhancedContextVAE(nn.Module):
    """
    Spatial Context-Encoding Variational Autoencoder (Spatial ceVAE).
    Integrates:
    - Spatial Encoder (128x128 -> 16x16x128)
    - Spatial Bottleneck (16x16x128 -> 8x16x16 spatial latents)
    - Spatial Decoder with Residual Edge Refinement (8x16x16 -> 128x128x1)
    - Dual-Pass Training (Clean Pass + Context-Inpainting Erased Pass)
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 128,
        base_channels: int = 32,
        latent_dim: int = 8,  # Spatial latent channels (8x16x16 = 2048 spatial latents)
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels, negative_slope=negative_slope)
        self.bottleneck = Bottleneck(in_channels=base_channels * 4, latent_channels=latent_dim)
        self.decoder = Decoder(
            latent_channels=latent_dim,
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
        # 1. Clean forward pass
        z_clean, mu_clean, logvar_clean = self.encode(x_clean)
        recon_clean = self.decode(z_clean)

        outputs = {
            "recon_clean": recon_clean,
            "mu": mu_clean,
            "logvar": logvar_clean,
            "z": z_clean,
        }

        # 2. Context-inpainting pass
        if x_masked is not None:
            z_masked, mu_masked, logvar_masked = self.encode(x_masked)
            recon_inpaint = self.decode(z_masked)
            outputs["recon_inpaint"] = recon_inpaint
            outputs["mu_masked"] = mu_masked
            outputs["logvar_masked"] = logvar_masked

        return outputs
