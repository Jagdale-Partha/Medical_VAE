"""
CLI Evaluation and Diagnostic Visualization Script for ceVAE+.
Usage:
    python evaluate.py --checkpoint checkpoints/best_cevae_model.pt --num_samples 5
"""

import argparse
from pathlib import Path
import torch

from src.config import default_cfg, ExperimentConfig
from src.data.dataset import get_uad_dataloaders
from src.models.cevae import EnhancedContextVAE
from src.engine.anomaly_scorer import DiagnosticAnomalyScorer
from src.metrics.evaluator import UADEvaluator
from src.visualize.plotting import plot_5panel_diagnostic


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ceVAE+ UAD on Pathological MRI Slices.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default=default_cfg.train.device, help="Compute device (cuda/cpu)")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Number of 5-panel figures to save")
    parser.add_argument("--output_dir", type=str, default=str(default_cfg.paths.results_dir), help="Results output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.train.device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting ceVAE+ Evaluation & Benchmark ===")
    print(f"Device: {cfg.train.device}")

    # Initialize model
    model = EnhancedContextVAE(
        in_channels=cfg.model.in_channels,
        image_size=cfg.model.image_size,
        base_channels=cfg.model.base_channels,
        latent_dim=cfg.model.latent_dim,
        negative_slope=cfg.model.leaky_relu_slope,
    ).to(cfg.train.device)

    # Load weights if checkpoint provided
    chk_path = Path(args.checkpoint)
    if chk_path.is_file():
        print(f"Loading checkpoint from: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=cfg.train.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Notice: No checkpoint provided at '{args.checkpoint}'. Running benchmark in demonstration mode.")

    # Data loader
    print("\n[1/3] Loading test benchmark (pathological slices + healthy controls)...")
    _, _, test_loader = get_uad_dataloaders(
        batch_size=8,
        image_size=cfg.model.image_size,
        num_workers=0,
        seed=42,
    )
    print(f"  Test batches: {len(test_loader)}")

    # Diagnostic scorer and evaluator
    print("\n[2/3] Executing dual-scoring autograd inference across test dataset...")
    scorer = DiagnosticAnomalyScorer(
        model=model,
        gaussian_sigma=cfg.scorer.gaussian_sigma,
        gaussian_kernel_size=cfg.scorer.gaussian_kernel_size,
    )
    evaluator = UADEvaluator(scorer=scorer, device=cfg.train.device)

    results = evaluator.evaluate_dataset(test_loader)

    print("\n=== Quantitative Benchmark Results ===")
    print(f"  Pixel-Level Metrics:")
    print(f"    - Pixel AUROC:                {results['pixel_auroc']:.4f}")
    print(f"    - Pixel AUPRC:                {results['pixel_auprc']:.4f}")
    print(f"    - Theoretical Max Dice:       {results['optimal_dice']:.4f} (at threshold {results['optimal_threshold']:.3f})")
    print(f"  Slice-Level Metrics:")
    print(f"    - Slice Classification AUROC: {results['slice_auroc']:.4f}")
    print(f"    - Slice Classification AUPRC: {results['slice_auprc']:.4f}")

    # Generate 5-panel diagnostic figures
    print(f"\n[3/3] Generating {args.num_visualizations} 5-panel clinical diagnostic figures...")
    patho_indices = [
        i for i in range(len(results["gt_masks"])) if results["gt_masks"][i].sum() > 0
    ][: args.num_visualizations]

    for rank, idx in enumerate(patho_indices):
        fig_path = out_dir / f"diagnostic_panel_patho_{rank+1}.png"
        plot_5panel_diagnostic(
            input_slice=results["inputs"][idx],
            reconstruction=results["reconstructions"][idx],
            residual=results["residuals"][idx],
            kl_saliency=results["kl_saliencies"][idx],
            anomaly_map=results["anomaly_maps"][idx],
            gt_mask=results["gt_masks"][idx],
            threshold=results["optimal_threshold"],
            title=f"ceVAE+ Anomaly Diagnostic - Case #{rank+1} (Optimal Dice: {results['optimal_dice']:.3f})",
            save_path=fig_path,
        )
        print(f"  Saved: {fig_path}")

    print(f"\nEvaluation complete. Diagnostic outputs stored in: {out_dir}")


if __name__ == "__main__":
    main()
