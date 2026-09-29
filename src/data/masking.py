"""
Dynamic Spatial Masking / Random Erasing Module for ceVAE+.
Implements context-encoding perturbation protocol:
\\tilde{x} = x \\odot (1 - M) with random rectangular boxes of size 16x16 to 32x32.
"""

from typing import Tuple
import torch
import torch.nn as nn


class RandomSpatialEraser(nn.Module):
    """
    Applies on-the-fly dynamic rectangular spatial erasing to input MRI slices.
    Generates binary mask M \\in {0, 1}^{1 \\times H \\times W} where 1 denotes masked/erased pixels.
    Output: perturbed image \\tilde{x} = x * (1 - M) and mask M.
    """

    def __init__(
        self,
        min_box_size: int = 16,
        max_box_size: int = 32,
        mask_value: float = 0.0,
        p_apply: float = 1.0,
    ):
        super().__init__()
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.mask_value = mask_value
        self.p_apply = p_apply

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, C, H, W) normalized to [0.0, 1.0]
        Returns:
            x_masked: Tensor of shape (B, C, H, W) with erased box
            mask: Binary tensor of shape (B, 1, H, W) where erased region = 1.0
        """
        B, C, H, W = x.shape
        device = x.device
        masks = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for b in range(B):
            if torch.rand(1).item() > self.p_apply:
                continue

            # Random bounding box dimensions within [min_box_size, max_box_size]
            box_h = torch.randint(self.min_box_size, self.max_box_size + 1, (1,)).item()
            box_w = torch.randint(self.min_box_size, self.max_box_size + 1, (1,)).item()

            # Ensure box fits within image boundaries
            top = torch.randint(0, max(1, H - box_h + 1), (1,)).item()
            left = torch.randint(0, max(1, W - box_w + 1), (1,)).item()

            masks[b, 0, top : top + box_h, left : left + box_w] = 1.0

        x_masked = x * (1.0 - masks) + self.mask_value * masks
        return x_masked, masks
