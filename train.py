"""
CLI Training Script for ceVAE+ Unsupervised Anomaly Detection in Brain MRI.
Usage:
    python train.py --epochs 30 --batch_size 16 --lr 1e-4
"""

import argparse
from pathlib import Path
import torch

from src.config import default_cfg, ExperimentConfig
from src.data.dataset import get_uad_dataloaders
from src.data.masking import RandomSpatialEraser
from src.models.cevae import EnhancedContextVAE
from src.losses.cevae_loss import CompositeCeVAELoss
from src.engine.trainer import CeVAETrainer
from src.visualize.plotting import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train ceVAE+ on Healthy Brain MRI Slices.")
    parser.add_argument("--epochs", type=int, default=default_cfg.train.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=default_cfg.train.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=default_cfg.train.learning_rate, help="Learning rate")
    parser.add_argument("--mse_weight", type=float, default=default_cfg.loss.mse_weight, help="MSE reconstruction weight")
    parser.add_argument("--l1_weight", type=float, default=default_cfg.loss.l1_weight, help="L1 reconstruction weight")
    parser.add_argument("--beta_kl", type=float, default=default_cfg.loss.beta_kl, help="KL loss weight beta")
    parser.add_argument("--device", type=str, default=default_cfg.train.device, help="Compute device (cuda/cpu)")
    parser.add_argument("--dataset", type=str, default="dataset_128", choices=["dataset_128", "ixi_t1", "axial_brain_mri", "real_brain_mri", "medmnist", "phantom"], help="Dataset source (dataset_128, ixi_t1, axial_brain_mri, real_brain_mri, medmnist, or phantom)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training slices (default: all)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum number of validation slices (default: all)")
    parser.add_argument("--spatial_vae", action="store_true", help="Train pure Spatial VAE (single-pass, 2x faster, razor-sharp edges)")
    parser.add_argument("--output_dir", type=str, default=str(default_cfg.paths.checkpoints_dir), help="Checkpoint directory")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.loss.mse_weight = args.mse_weight
    cfg.loss.l1_weight = args.l1_weight
    cfg.loss.beta_kl = args.beta_kl
    cfg.train.device = args.device
    if args.spatial_vae:
        cfg.masking.p_apply = 0.0
    cfg.paths.checkpoints_dir = Path(args.output_dir)
    cfg.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting ceVAE+ Training ===")
    print(f"Device: {cfg.train.device} | Epochs: {cfg.train.epochs} | Batch Size: {cfg.train.batch_size} | Dataset: {args.dataset}")
    print(f"Loss formulation: {cfg.loss.mse_weight} * MSE + {cfg.loss.l1_weight} * L1 + beta ({cfg.loss.beta_kl}) * KL")

    # Data loaders
    print(f"\n[1/4] Preparing normative healthy dataset splits ({args.dataset})...")
    train_loader, val_loader, _ = get_uad_dataloaders(
        dataset_source=args.dataset,
        batch_size=cfg.train.batch_size,
        image_size=cfg.model.image_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
    )
    print(f"  Train healthy batches: {len(train_loader)} | Val healthy batches: {len(val_loader)}")

    # Model definition
    print("\n[2/4] Initializing ceVAE+ model architecture...")
    model = EnhancedContextVAE(
        in_channels=cfg.model.in_channels,
        image_size=cfg.model.image_size,
        base_channels=cfg.model.base_channels,
        latent_dim=cfg.model.latent_dim,
        negative_slope=cfg.model.leaky_relu_slope,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model instantiated successfully. Trainable parameters: {num_params:,}")

    # Loss engine & Perturbation module
    print("\n[3/4] Initializing loss engine and dynamic spatial eraser...")
    loss_fn = CompositeCeVAELoss(
        mse_weight=cfg.loss.mse_weight,
        l1_weight=cfg.loss.l1_weight,
        bce_weight=cfg.loss.bce_weight,
        ssim_weight=cfg.loss.ssim_weight,
        edge_weight=cfg.loss.edge_weight,
        beta_kl=cfg.loss.beta_kl,
        clean_weight=cfg.loss.clean_loss_weight,
        inpaint_weight=cfg.loss.inpaint_loss_weight,
    )
    eraser = RandomSpatialEraser(
        min_box_size=cfg.masking.min_box_size,
        max_box_size=cfg.masking.max_box_size,
        mask_value=cfg.masking.mask_value,
        p_apply=cfg.masking.p_apply,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    trainer = CeVAETrainer(
        model=model,
        loss_fn=loss_fn,
        eraser=eraser,
        optimizer=optimizer,
        config=cfg,
    )

    # Training
    print("\n[4/4] Commencing model training...")
    chk_file = cfg.paths.checkpoints_dir / "best_cevae_model.pt"
    history = trainer.fit(train_loader, val_loader, checkpoint_path=chk_file)

    # Save training curves
    curve_file = cfg.paths.results_dir / "training_curves.png"
    plot_training_curves(history, save_path=curve_file)
    print(f"\nTraining completed!")
    print(f"Best model checkpoint saved to: {chk_file}")
    print(f"Training curves saved to: {curve_file}")


if __name__ == "__main__":
    main()
