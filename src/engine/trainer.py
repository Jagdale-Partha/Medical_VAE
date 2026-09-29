"""
Trainer Module for ceVAE+.
Handles training loop, validation, beta-annealing warmup, gradient clipping,
and model checkpointing.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cevae import EnhancedContextVAE
from src.data.masking import RandomSpatialEraser
from src.losses.cevae_loss import CompositeCeVAELoss
from src.config import ExperimentConfig


class CeVAETrainer:
    """
    Manages end-to-end training and validation for ceVAE+.
    """

    def __init__(
        self,
        model: EnhancedContextVAE,
        loss_fn: CompositeCeVAELoss,
        eraser: RandomSpatialEraser,
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.eraser = eraser
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.train.device)
        self.model.to(self.device)

        self.history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        # Beta annealing warmup
        warmup_epochs = self.config.loss.kl_warmup_epochs
        if epoch < warmup_epochs:
            current_beta = self.config.loss.beta_kl * ((epoch + 1) / warmup_epochs)
        else:
            current_beta = self.config.loss.beta_kl

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.train.epochs} [Train]", leave=False)
        for batch in pbar:
            x_clean = batch["image"].to(self.device)

            # Apply dynamic random spatial erasing
            x_masked, _ = self.eraser(x_clean)

            self.optimizer.zero_grad()

            # Dual forward pass: clean + masked inpainting
            outputs = self.model(x_clean=x_clean, x_masked=x_masked)

            # Composite loss
            loss_dict = self.loss_fn(
                target_clean=x_clean,
                model_outputs=outputs,
                current_beta=current_beta,
            )

            loss = loss_dict["loss_total"]
            loss.backward()

            # Gradient clipping to stabilize dense bottleneck
            if self.config.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_recon += loss_dict["loss_recon_clean"].item()
            total_kl += loss_dict["loss_kl"].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{loss_dict['loss_recon_clean'].item():.4f}",
                "kl": f"{loss_dict['loss_kl'].item():.4f}",
                "beta": f"{current_beta:.5f}",
            })

        avg_loss = total_loss / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_kl = total_kl / max(1, n_batches)

        return {"loss": avg_loss, "recon": avg_recon, "kl": avg_kl, "beta": current_beta}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in dataloader:
            x_clean = batch["image"].to(self.device)
            outputs = self.model(x_clean=x_clean)

            loss_dict = self.loss_fn(
                target_clean=x_clean,
                model_outputs=outputs,
                current_beta=self.config.loss.beta_kl,
            )

            total_loss += loss_dict["loss_total"].item()
            total_recon += loss_dict["loss_recon_clean"].item()
            total_kl += loss_dict["loss_kl"].item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_kl = total_kl / max(1, n_batches)

        return {"val_loss": avg_loss, "val_recon": avg_recon, "val_kl": avg_kl}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, list]:
        """
        Executes full training across epochs.
        """
        best_val_loss = float("inf")
        chk_path = checkpoint_path or (self.config.paths.checkpoints_dir / "best_cevae_model.pt")

        for epoch in range(self.config.train.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_recon"].append(train_metrics["recon"])
            self.history["train_kl"].append(train_metrics["kl"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_recon"].append(val_metrics["val_recon"])
            self.history["val_kl"].append(val_metrics["val_kl"])

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": self.config,
                }, chk_path)

        return self.history
