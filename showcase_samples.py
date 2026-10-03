"""
showcase_samples.py - Showcase 5 Healthy and 5 Unhealthy Brain MRI Samples
Evaluates ceVAE+ reconstructions, residuals, autograd KL saliency, and anomaly maps
on 5 healthy normative brain slices and 5 pathological brain tumor slices.
Saves comprehensive multi-panel figures to results/.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from src.config import default_cfg, ExperimentConfig
from src.models.cevae import EnhancedContextVAE
from src.engine.anomaly_scorer import DiagnosticAnomalyScorer
from src.data.dataset import get_uad_dataloaders


def create_showcase_grid(
    samples_list,
    title: str,
    output_path: Path,
    is_pathological: bool = False,
):
    """
    Renders a 5-row by 5-column clinical diagnostic matrix:
    Columns: [1] Original Input Slice, [2] ceVAE+ Reconstruction, [3] Residual |x - x_hat|,
             [4] KL Saliency Map, [5] Dual-Scored Anomaly Heatmap
    """
    num_samples = len(samples_list)
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3.0 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        r"Original Input $x$",
        r"ceVAE+ Reconstruction $\hat{x}$",
        r"Residual $|x - \hat{x}|$",
        r"KL Saliency $|\partial \mathcal{L}_{KL} / \partial x|$",
        r"Dual-Scored Anomaly Map",
    ]

    for c, c_title in enumerate(col_titles):
        axes[0, c].set_title(c_title, fontsize=12, pad=10, fontweight="bold")

    for r, item in enumerate(samples_list):
        inp = item["input"]
        rec = item["recon"]
        res = item["residual"]
        kl = item["kl"]
        anom = item["anomaly"]
        gt = item.get("gt", None)
        lbl = item.get("label_name", f"Sample #{r+1}")

        # Column 0: Original Input
        axes[r, 0].imshow(inp, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(lbl, fontsize=11, fontweight="bold", labelpad=8)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        # Column 1: Reconstruction
        axes[r, 1].imshow(rec, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].axis("off")

        # Column 2: Residual |x - \hat{x}|
        im_res = axes[r, 2].imshow(res, cmap="plasma", vmin=0, vmax=max(0.5, float(res.max())))
        axes[r, 2].axis("off")

        # Column 3: KL Saliency Map
        im_kl = axes[r, 3].imshow(kl, cmap="viridis", vmin=0, vmax=max(0.01, float(kl.max())))
        axes[r, 3].axis("off")

        # Column 4: Dual-Scored Anomaly Heatmap with GT contour
        axes[r, 4].imshow(inp, cmap="gray", alpha=0.6, vmin=0, vmax=1)
        im_anom = axes[r, 4].imshow(anom, cmap="inferno", alpha=0.65, vmin=0, vmax=max(0.5, float(anom.max())))
        if gt is not None and gt.sum() > 0:
            axes[r, 4].contour(gt > 0.5, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[r, 4].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved showcase gallery to: {output_path}", flush=True)


def run_showcase(
    checkpoint_path: str = "checkpoints/best_cevae_model.pt",
    device: str = "cpu",
    output_dir: str = "results",
):
    chk = Path(checkpoint_path)
    if device == "cuda" and not torch.cuda.is_available():
        device_obj = torch.device("cpu")
    elif device == "cuda":
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = device_obj
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ceVAE+ Diagnostic Showcase (5 Healthy + 5 Unhealthy) ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {chk}", flush=True)

    cfg = ExperimentConfig()
    model = EnhancedContextVAE(
        in_channels=cfg.model.in_channels,
        image_size=cfg.model.image_size,
        base_channels=cfg.model.base_channels,
        latent_dim=cfg.model.latent_dim,
        negative_slope=cfg.model.leaky_relu_slope,
    ).to(device)

    if chk.is_file():
        print(f"Loading checkpoint weights from {chk}...", flush=True)
        checkpoint = torch.load(chk, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", float("nan"))
        print(f"[OK] Checkpoint loaded successfully! (Trained Epoch: {epoch_info}, Best Val Loss: {val_loss:.4f})", flush=True)
    else:
        print(f"Notice: Checkpoint {chk} not found. Running with initialized weights for test validation.", flush=True)

    model.eval()

    scorer = DiagnosticAnomalyScorer(
        model=model,
        gaussian_sigma=cfg.scorer.gaussian_sigma,
        gaussian_kernel_size=cfg.scorer.gaussian_kernel_size,
    )

    # Load test data
    print("Loading test cohort (normative healthy + clinical brain tumors)...", flush=True)
    _, _, test_loader = get_uad_dataloaders(
        dataset_source="ixi_t1",
        batch_size=16,
        image_size=cfg.model.image_size,
        num_workers=0,
        seed=42,
    )

    healthy_samples = []
    unhealthy_samples = []

    # Gather test slices - only score when needed
    for batch in test_loader:
        imgs = batch["image"]
        msks = batch.get("mask", torch.zeros_like(imgs))
        lbls = batch.get("label", torch.zeros(len(imgs)))

        for i in range(len(imgs)):
            label = int(lbls[i].item())
            if label == 0 and len(healthy_samples) >= 5:
                continue
            if label == 1 and len(unhealthy_samples) >= 5:
                continue

            x_tensor = imgs[i:i+1].to(device)
            gt_mask = msks[i, 0].cpu().numpy()

            score_out = scorer.score_slice(x_tensor)
            inp_np = x_tensor.squeeze().detach().cpu().numpy()
            rec_np = score_out["reconstruction"].squeeze().detach().cpu().numpy()
            res_np = score_out["residual"].squeeze().detach().cpu().numpy()
            kl_np = score_out["kl_saliency"].squeeze().detach().cpu().numpy()
            anom_np = score_out["anomaly_map"].squeeze().detach().cpu().numpy()

            sample_entry = {
                "input": inp_np,
                "recon": rec_np,
                "residual": res_np,
                "kl": kl_np,
                "anomaly": anom_np,
                "gt": gt_mask,
            }

            if label == 0 and len(healthy_samples) < 5:
                sample_entry["label_name"] = f"Healthy Control #{len(healthy_samples) + 1}"
                healthy_samples.append(sample_entry)
                print(f"  Processed Healthy Control #{len(healthy_samples)}/5", flush=True)
            elif label == 1 and len(unhealthy_samples) < 5:
                sample_entry["label_name"] = f"Tumor Pathology #{len(unhealthy_samples) + 1}"
                unhealthy_samples.append(sample_entry)
                print(f"  Processed Tumor Pathology #{len(unhealthy_samples)}/5", flush=True)

            if len(healthy_samples) >= 5 and len(unhealthy_samples) >= 5:
                break

        if len(healthy_samples) >= 5 and len(unhealthy_samples) >= 5:
            break

    print(f"[OK] Collected {len(healthy_samples)} healthy slices and {len(unhealthy_samples)} unhealthy slices.", flush=True)

    # Render Healthy Gallery
    healthy_path = out_dir / "showcase_5_healthy_examples.png"
    create_showcase_grid(
        healthy_samples,
        title="ceVAE+ Normative Diagnostic: 5 Healthy Brain MRI Slices\n(Clean Reconstruction & Near-Zero Anomaly Signal)",
        output_path=healthy_path,
        is_pathological=False,
    )

    # Render Unhealthy Gallery
    unhealthy_path = out_dir / "showcase_5_unhealthy_examples.png"
    create_showcase_grid(
        unhealthy_samples,
        title="ceVAE+ Clinical Anomaly Detection: 5 Brain Tumor Slices\n(Normative Healing, High Residual & Anomaly Heatmap Localization)",
        output_path=unhealthy_path,
        is_pathological=True,
    )

    # Combined Side-by-Side Summary
    combined_samples = healthy_samples + unhealthy_samples
    combined_path = out_dir / "showcase_10_samples_summary.png"
    create_showcase_grid(
        combined_samples,
        title="ceVAE+ Full Diagnostic Cohort: 5 Healthy Normative Controls (Top) vs 5 Brain Tumor Cases (Bottom)",
        output_path=combined_path,
        is_pathological=True,
    )

    print(f"\n=======================================================", flush=True)
    print(f"[OK] ALL 10 SHOWCASE EXAMPLES GENERATED SUCCESSFULLY!", flush=True)
    print(f"  1. Healthy Showcase:   {healthy_path}", flush=True)
    print(f"  2. Unhealthy Showcase: {unhealthy_path}", flush=True)
    print(f"  3. Combined Summary:   {combined_path}", flush=True)
    print(f"=======================================================\n", flush=True)
    return healthy_path, unhealthy_path, combined_path


if __name__ == "__main__":
    chk_arg = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_cevae_model.pt"
    run_showcase(checkpoint_path=chk_arg)
