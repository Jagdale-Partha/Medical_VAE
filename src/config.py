"""
Configuration Module for ceVAE+ Unsupervised Anomaly Detection (UAD) in Brain MRI.
Centralizes hyperparameters, architecture dimensions, perturbation protocols, and loss weights.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    in_channels: int = 1
    image_size: int = 128
    base_channels: int = 32  # 32 -> 64 -> 128
    latent_dim: int = 16  # Spatial latent channels (16 x 16 x 16)
    flattened_dim: int = 16 * 16 * 16  # 4096
    leaky_relu_slope: float = 0.2


@dataclass
class MaskingConfig:
    min_box_size: int = 16
    max_box_size: int = 32
    mask_value: float = 0.0  # Zero-out erased spatial context
    p_apply: float = 1.0  # Probability of applying spatial masking during training


@dataclass
class LossConfig:
    mse_weight: float = 1.0
    l1_weight: float = 0.5
    bce_weight: float = 0.0
    ssim_weight: float = 0.0
    edge_weight: float = 0.0
    beta_kl: float = 0.001
    clean_loss_weight: float = 1.0
    inpaint_loss_weight: float = 0.5
    kl_warmup_epochs: int = 3  # Linear warmup to prevent posterior collapse


@dataclass
class AnomalyScorerConfig:
    gaussian_sigma: float = 1.5
    gaussian_kernel_size: int = 7
    threshold_steps: int = 200  # Resolution for optimal theoretical Dice search


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 0  # Safe default on Windows


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    checkpoints_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scorer: AnomalyScorerConfig = field(default_factory=AnomalyScorerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Default global configuration instance
default_cfg = ExperimentConfig()
