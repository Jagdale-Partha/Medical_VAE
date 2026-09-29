"""
Rapid Self-Test and Smoke Test for ceVAE+ Architecture & Pipeline.
Executes an end-to-end forward/backward autograd sanity pass in < 5 seconds.
"""

import sys
from pathlib import Path
import torch

from src.config import default_cfg
from src.data.masking import RandomSpatialEraser
from src.data.dataset import HighFidelityBrainPhantom
from src.models.cevae import EnhancedContextVAE
from src.losses.cevae_loss import CompositeCeVAELoss
from src.engine.anomaly_scorer import DiagnosticAnomalyScorer
from src.visualize.plotting import plot_5panel_diagnostic


def run_smoke_test():
    print("=" * 60)
    print("Running ceVAE+ Pipeline Sanity and Smoke Test")
    print("=" * 60)

    # 1. Dataset & Phantom Check
    print("\n[Step 1] Synthesizing healthy & pathological brain MRI slices...")
    img_h, mask_h = HighFidelityBrainPhantom.generate_slice(128, is_pathological=False, seed=42)
    img_p, mask_p = HighFidelityBrainPhantom.generate_slice(128, is_pathological=True, seed=101)
    assert img_h.shape == (128, 128), f"Bad shape: {img_h.shape}"
    assert mask_h.sum() == 0.0, "Healthy slice mask must be all zero"
    assert mask_p.sum() > 0.0, "Pathological slice mask must have positive lesion pixels"
    print("  -> Brain phantom generation verified.")

    # 2. Dynamic Spatial Eraser Check
    print("\n[Step 2] Testing on-the-fly random spatial eraser...")
    eraser = RandomSpatialEraser(min_box_size=16, max_box_size=32)
    x = torch.from_numpy(img_h).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
    x_masked, mask = eraser(x)
    assert x_masked.shape == (1, 1, 128, 128)
    assert mask.shape == (1, 1, 128, 128)
    assert mask.sum() >= (16 * 16), "Mask area smaller than minimum box"
    print(f"  -> Dynamic spatial eraser verified. Erased pixels: {int(mask.sum().item())}")

    # 3. Model Architecture & Forward Pass Check
    print("\n[Step 3] Initializing ceVAE+ model architecture...")
    model = EnhancedContextVAE(
        in_channels=1,
        image_size=128,
        base_channels=32,
        latent_dim=128,
        negative_slope=0.2,
    )
    outputs = model(x_clean=x, x_masked=x_masked)
    assert "recon_clean" in outputs and "recon_inpaint" in outputs
    assert outputs["recon_clean"].shape == (1, 1, 128, 128)
    assert outputs["z"].shape == (1, 128)
    print("  -> Model forward pass verified (Clean + Inpainting paths).")

    # 4. Loss Engine Check
    print("\n[Step 4] Computing composite loss (0.8 L1 + 0.2 (1-SSIM) + beta * KL)...")
    loss_fn = CompositeCeVAELoss(l1_weight=0.8, ssim_weight=0.2, beta_kl=0.001)
    loss_dict = loss_fn(target_clean=x, model_outputs=outputs)
    assert not torch.isnan(loss_dict["loss_total"]), "NaN encountered in total loss"
    assert not torch.isnan(loss_dict["loss_kl"]), "NaN encountered in KL loss"
    print(f"  -> Loss engine verified. Total Loss: {loss_dict['loss_total'].item():.4f} | SSIM Loss: {loss_dict['loss_ssim'].item():.4f}")

    # 5. Autograd KL Saliency & Anomaly Scorer Check
    print("\n[Step 5] Computing input-level autograd KL saliency & dual-scoring...")
    scorer = DiagnosticAnomalyScorer(model=model, gaussian_sigma=1.5, gaussian_kernel_size=7)
    x_test = torch.from_numpy(img_p).unsqueeze(0).unsqueeze(0)
    score_outputs = scorer.score_slice(x_test)
    assert score_outputs["kl_saliency"].shape == (1, 1, 128, 128)
    assert score_outputs["anomaly_map"].shape == (1, 1, 128, 128)
    print("  -> Dual-scoring and input-gradient autograd saliency verified.")

    # 6. Visualization Check
    print("\n[Step 6] Rendering 5-panel clinical diagnostic figure...")
    test_out_dir = Path("results")
    test_out_dir.mkdir(exist_ok=True)
    fig_path = test_out_dir / "smoke_test_diagnostic.png"
    plot_5panel_diagnostic(
        input_slice=img_p,
        reconstruction=score_outputs["reconstruction"][0].cpu().numpy(),
        residual=score_outputs["residual"][0].cpu().numpy(),
        kl_saliency=score_outputs["kl_saliency"][0].cpu().numpy(),
        anomaly_map=score_outputs["anomaly_map"][0].cpu().numpy(),
        gt_mask=mask_p,
        title="Smoke Test Diagnostic Pass",
        save_path=fig_path,
    )
    assert fig_path.exists(), "Diagnostic figure was not saved"
    print(f"  -> 5-panel diagnostic plot saved to: {fig_path}")

    print("\n" + "=" * 60)
    print("All ceVAE+ components passed unit verification flawlessly!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
