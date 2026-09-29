"""
Visualization and Diagnostic Plotting Engine for ceVAE+.
Implements:
1. 5-panel clinical diagnostic figure:
   [1: Input x, 2: Reconstruction \\hat{x}, 3: Residual |x - \\hat{x}|, 4: KL Saliency |\\partial L_KL / \\partial x|, 5: Predicted Mask vs Ground Truth overlay]
2. Training loss curves (Reconstruction, KL divergence, Total Loss)
3. ROC & PR diagnostic curves for pixel & slice levels
"""

from typing import Optional, List
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI / server
import matplotlib.pyplot as plt


def plot_5panel_diagnostic(
    input_slice: np.ndarray,
    reconstruction: np.ndarray,
    residual: np.ndarray,
    kl_saliency: np.ndarray,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    title: str = "ceVAE+ Anomaly Localization Diagnostic",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Renders 5-panel diagnostic Matplotlib figure:
    1. Input slice x (axial T2-FLAIR)
    2. Model normative reconstruction \\hat{x}
    3. Spatial residual |x - \\hat{x}|
    4. Input-level KL gradient saliency |\\partial L_KL / \\partial x|
    5. Predicted anomaly mask vs Ground Truth overlay (Green: GT, Red: Pred, Yellow: Overlap)
    """
    # Squeeze channel dimensions if present
    x_img = np.squeeze(input_slice)
    x_rec = np.squeeze(reconstruction)
    x_res = np.squeeze(residual)
    x_kl = np.squeeze(kl_saliency)
    x_anom = np.squeeze(anomaly_map)
    x_gt = np.squeeze(gt_mask)

    # Normalize anomaly map for binary segmentation thresholding
    a_min, a_max = x_anom.min(), x_anom.max()
    norm_anom = (x_anom - a_min) / (a_max - a_min + 1e-8)
    pred_bin = (norm_anom >= threshold).astype(np.float32)

    # Build RGB overlay for Panel 5
    # Base grayscale image
    base_rgb = np.repeat(x_img[..., np.newaxis], 3, axis=-1)
    overlay_rgb = base_rgb.copy()

    # Red: Predicted Lesion
    # Green: Ground Truth Lesion
    # Yellow (Red + Green): True Positive Overlap
    pred_mask = pred_bin > 0.5
    gt_mask_bin = x_gt > 0.5

    overlay_rgb[pred_mask, 0] = 0.95  # Red channel
    overlay_rgb[pred_mask, 1] = 0.20
    overlay_rgb[pred_mask, 2] = 0.20

    overlay_rgb[gt_mask_bin, 1] = 0.95  # Green channel
    if np.any(pred_mask & gt_mask_bin):
        overlay_rgb[pred_mask & gt_mask_bin, 0] = 1.0  # Yellow
        overlay_rgb[pred_mask & gt_mask_bin, 1] = 0.90
        overlay_rgb[pred_mask & gt_mask_bin, 2] = 0.0

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    # Panel 1: Input Slice
    im0 = axes[0].imshow(x_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("(1) Input T2-FLAIR Slice (x)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Reconstruction
    im1 = axes[1].imshow(x_rec, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(r"(2) Normative Recon ($\hat{x}$)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Spatial Residual
    im2 = axes[2].imshow(x_res, cmap="inferno")
    axes[2].set_title(r"(3) Residual $|x - \hat{x}|$", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: KL Saliency
    im3 = axes[3].imshow(x_kl, cmap="viridis")
    axes[3].set_title(r"(4) KL Saliency $|\nabla_x \mathcal{L}_{KL}|$", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: Mask Overlay
    axes[5-1].imshow(overlay_rgb)
    axes[4].set_title("(5) Pred (Red) vs GT (Green)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.05)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots training and validation loss progression over epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    # Total Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Total", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Total", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Total Composite Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reconstruction Loss
    axes[1].plot(epochs, history["train_recon"], label="Train Recon (0.8 L1 + 0.2 SSIM)", color="#2ca02c", lw=2)
    axes[1].plot(epochs, history["val_recon"], label="Val Recon", color="#d62728", lw=2, linestyle="--")
    axes[1].set_title("Reconstruction Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # KL Divergence
    axes[2].plot(epochs, history["train_kl"], label="Train KL Divergence", color="#9467bd", lw=2)
    axes[2].plot(epochs, history["val_kl"], label="Val KL Divergence", color="#8c564b", lw=2, linestyle="--")
    axes[2].set_title(r"Latent Regularization ($\mathcal{L}_{KL}$)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
