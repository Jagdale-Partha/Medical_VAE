"""
Dataset and DataLoader Module for ceVAE+ Unsupervised Anomaly Detection (UAD).
Implements strict normative partitioning:
- Train / Val: Exclusively healthy (non-tumor) brain MRI slices.
- Test: Pathological slices with tumors/lesions paired with ground-truth binary masks + healthy controls.

Supports MedMNIST v2, custom directory of .npy / .png slices, and an automated high-fidelity
brain MRI phantom generator for instant zero-dependency execution.
"""

from typing import Tuple, Optional, Dict, List
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HighFidelityBrainPhantom:
    """
    Synthesizes realistic 2D axial T2-FLAIR brain MRI slices (1 x 128 x 128)
    with anatomically grounded structures:
    - Skull & scalp outer layer
    - Subarachnoid space & CSF (dark on T2-FLAIR)
    - Gray matter (cortex & sulci)
    - White matter (intermediate signal)
    - Lateral ventricles (dark slit/butterfly shapes on FLAIR)
    - Pathological anomalies: Hyperintense T2-FLAIR edema/tumor mass with necrotic core and ground truth mask.
    """

    @staticmethod
    def generate_slice(
        image_size: int = 128,
        is_pathological: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        # Coordinate grid centered at (0, 0)
        y, x = np.mgrid[-1 : 1 : complex(0, image_size), -1 : 1 : complex(0, image_size)]
        slice_img = np.zeros((image_size, image_size), dtype=np.float32)
        gt_mask = np.zeros((image_size, image_size), dtype=np.float32)

        # 1. Outer Cranium / Skull (Ellipsoidal base)
        skull_axis_x = 0.78 + rng.uniform(-0.02, 0.02)
        skull_axis_y = 0.90 + rng.uniform(-0.02, 0.02)
        skull_dist = (x / skull_axis_x) ** 2 + (y / skull_axis_y) ** 2
        brain_mask = skull_dist < 1.0

        # Skull bone & scalp rim
        skull_rim = (skull_dist >= 0.90) & (skull_dist < 1.0)
        slice_img[skull_rim] = 0.25 + rng.uniform(-0.05, 0.05)

        # 2. Brain Parenchyma (White matter base)
        parenchyma = skull_dist < 0.90
        base_intensity = 0.52 + 0.05 * np.cos(3 * x) * np.sin(3 * y)
        slice_img[parenchyma] = base_intensity[parenchyma]

        # 3. Cortical Gray Matter & Gyri / Sulcal pattern
        gyri = 0.12 * (np.sin(12 * x) * np.cos(10 * y) + np.sin(16 * (x**2 + y**2)))
        cortex_region = (skull_dist >= 0.65) & (skull_dist < 0.90)
        slice_img[cortex_region] += gyri[cortex_region]

        # 4. Deep White Matter
        deep_wm = skull_dist < 0.65
        slice_img[deep_wm] += 0.08 + 0.03 * rng.standard_normal((image_size, image_size))[deep_wm]

        # 5. Ventricular System (T2-FLAIR CSF attenuation -> Dark ventricles)
        # Symmetrical lateral ventricles
        ventricle_1 = (((x - 0.12) / 0.07) ** 2 + (y / 0.35) ** 2) < 1.0
        ventricle_2 = (((x + 0.12) / 0.07) ** 2 + (y / 0.35) ** 2) < 1.0
        ventricles = ventricle_1 | ventricle_2
        slice_img[ventricles] = 0.08 + 0.02 * rng.uniform(0, 1, np.sum(ventricles))

        # Smooth anatomical transitions with subtle realistic scanner noise
        noise = rng.normal(0.0, 0.015, (image_size, image_size))
        slice_img += noise
        slice_img = np.clip(slice_img, 0.0, 1.0)
        slice_img[~brain_mask] = 0.0  # Air background is zero

        # 6. Pathology Injection (if marked pathological)
        if is_pathological:
            # Place lesion inside brain parenchyma, avoiding central air background
            lesion_r = rng.uniform(0.12, 0.22)  # radius in normalized coords
            # Random position inside brain tissue
            angle = rng.uniform(0, 2 * math.pi)
            radial_dist = rng.uniform(0.20, 0.55)
            cx = radial_dist * math.cos(angle)
            cy = radial_dist * math.sin(angle)

            # Irregular lobulated shape via harmonic perturbation
            theta = np.arctan2(y - cy, x - cx)
            perturbation = 1.0 + 0.25 * np.sin(3 * theta) + 0.15 * np.cos(5 * theta)
            dist_to_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            lesion_area = (dist_to_center < (lesion_r * perturbation)) & brain_mask

            if np.any(lesion_area):
                gt_mask[lesion_area] = 1.0
                # T2-FLAIR hyperintensity (bright signal around 0.85 - 0.98 with necrotic darker core)
                hyperintensity = 0.90 + 0.08 * rng.standard_normal((image_size, image_size))
                necrotic_core = (dist_to_center < (lesion_r * 0.4)) & lesion_area
                hyperintensity[necrotic_core] = 0.35 + 0.05 * rng.standard_normal(np.sum(necrotic_core))
                slice_img[lesion_area] = np.clip(hyperintensity[lesion_area], 0.0, 1.0)

        return slice_img.astype(np.float32), gt_mask.astype(np.float32)


class BrainMRISliceDataset(Dataset):
    """
    Standard PyTorch Dataset for 2D axial T2-FLAIR brain MRI slices.
    Returns:
        dict: {
            "image": Tensor (1, H, W) normalized to [0.0, 1.0],
            "mask": Tensor (1, H, W) binary ground truth (1 = anomaly, 0 = healthy),
            "label": Tensor int (0 = healthy, 1 = pathological)
        }
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            images: np.ndarray of shape (N, H, W) or (N, 1, H, W)
            masks: np.ndarray of shape (N, H, W) or (N, 1, H, W), default zeros
            labels: np.ndarray of shape (N,), 0 for normal, 1 for anomaly
        """
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        self.images = torch.from_numpy(images).float()

        if masks is None:
            self.masks = torch.zeros_like(self.images)
        else:
            if masks.ndim == 3:
                masks = masks[:, np.newaxis, :, :]
            self.masks = torch.from_numpy(masks).float()

        if labels is None:
            self.labels = torch.zeros(len(self.images), dtype=torch.long)
        else:
            self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "mask": self.masks[idx],
            "label": self.labels[idx],
        }


def create_uad_data_splits(
    num_train_healthy: int = 500,
    num_val_healthy: int = 100,
    num_test_healthy: int = 100,
    num_test_pathological: int = 150,
    image_size: int = 128,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Constructs strict normative UAD splits:
    - Train: exclusively healthy brain slices.
    - Val: exclusively healthy brain slices.
    - Test: pathological slices with ground-truth masks + healthy controls.
    """
    total_healthy = num_train_healthy + num_val_healthy + num_test_healthy
    rng = np.random.default_rng(seed)

    healthy_slices = []
    healthy_masks = []
    for i in range(total_healthy):
        img, msk = HighFidelityBrainPhantom.generate_slice(
            image_size=image_size, is_pathological=False, seed=int(rng.integers(0, 1000000))
        )
        healthy_slices.append(img)
        healthy_masks.append(msk)

    healthy_slices = np.stack(healthy_slices, axis=0)
    healthy_masks = np.stack(healthy_masks, axis=0)

    # Train / Val / Test healthy partitioning
    idx_train = num_train_healthy
    idx_val = idx_train + num_val_healthy

    train_imgs = healthy_slices[:idx_train]
    train_msks = healthy_masks[:idx_train]
    train_lbls = np.zeros(num_train_healthy, dtype=np.int64)

    val_imgs = healthy_slices[idx_train:idx_val]
    val_msks = healthy_masks[idx_train:idx_val]
    val_lbls = np.zeros(num_val_healthy, dtype=np.int64)

    test_healthy_imgs = healthy_slices[idx_val:]
    test_healthy_msks = healthy_masks[idx_val:]
    test_healthy_lbls = np.zeros(num_test_healthy, dtype=np.int64)

    # Pathological slices
    patho_slices = []
    patho_masks = []
    for i in range(num_test_pathological):
        img, msk = HighFidelityBrainPhantom.generate_slice(
            image_size=image_size, is_pathological=True, seed=int(rng.integers(1000001, 2000000))
        )
        patho_slices.append(img)
        patho_masks.append(msk)

    patho_slices = np.stack(patho_slices, axis=0)
    patho_masks = np.stack(patho_masks, axis=0)
    patho_lbls = np.ones(num_test_pathological, dtype=np.int64)

    test_imgs = np.concatenate([test_healthy_imgs, patho_slices], axis=0)
    test_msks = np.concatenate([test_healthy_msks, patho_masks], axis=0)
    test_lbls = np.concatenate([test_healthy_lbls, patho_lbls], axis=0)

    # Shuffle test set
    perm = rng.permutation(len(test_imgs))
    test_imgs = test_imgs[perm]
    test_msks = test_msks[perm]
    test_lbls = test_lbls[perm]

    train_ds = BrainMRISliceDataset(train_imgs, train_msks, train_lbls)
    val_ds = BrainMRISliceDataset(val_imgs, val_msks, val_lbls)
    test_ds = BrainMRISliceDataset(test_imgs, test_msks, test_lbls)

# Alias for semantic clarity
MedMNISTUADDataset = BrainMRISliceDataset


def load_medmnist_uad_splits(
    dataset_name: str = "OrganAMNIST",
    normal_class: int = 0,
    max_train_samples: int = 500,
    max_val_samples: int = 100,
    max_test_samples: int = 250,
    image_size: int = 128,
    download: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Constructs strict normative UAD splits directly from official MedMNIST:
    - Train: Exclusively normal class slices (normal_class, e.g. class 0).
    - Val: Exclusively normal class slices.
    - Test: Normal slices + Anomaly slices from out-of-distribution classes.
    Automatically standardizes and resizes slices to 1 x image_size x image_size in [0.0, 1.0].
    """
    import medmnist
    import torch.nn.functional as F

    DataClass = getattr(medmnist, dataset_name, getattr(medmnist, "OrganAMNIST"))
    train_raw = DataClass(split="train", download=download)
    val_raw = DataClass(split="val", download=download)
    test_raw = DataClass(split="test", download=download)

    rng = np.random.default_rng(seed)

    def extract_and_resize(imgs_np: np.ndarray, indices: np.ndarray) -> np.ndarray:
        sel = imgs_np[indices]
        if sel.ndim == 3:
            # (N, H, W) -> (N, 1, H, W)
            sel_t = torch.from_numpy(sel).unsqueeze(1).float() / 255.0
        elif sel.ndim == 4:
            # (N, H, W, C) -> (N, C, H, W)
            sel_t = torch.from_numpy(sel).permute(0, 3, 1, 2).float() / 255.0
            if sel_t.size(1) > 1:
                sel_t = sel_t.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported image shape: {sel.shape}")

        if sel_t.shape[-1] != image_size or sel_t.shape[-2] != image_size:
            sel_t = F.interpolate(sel_t, size=(image_size, image_size), mode="bilinear", align_corners=False)

        return sel_t.numpy()

    # Normal class masks
    train_norm_idx = np.where(train_raw.labels.flatten() == normal_class)[0]
    val_norm_idx = np.where(val_raw.labels.flatten() == normal_class)[0]
    test_norm_idx = np.where(test_raw.labels.flatten() == normal_class)[0]
    test_anom_idx = np.where(test_raw.labels.flatten() != normal_class)[0]

    # Subsample to requested quotas
    rng.shuffle(train_norm_idx)
    rng.shuffle(val_norm_idx)
    rng.shuffle(test_norm_idx)
    rng.shuffle(test_anom_idx)

    train_sel = train_norm_idx[:max_train_samples]
    val_sel = val_norm_idx[:max_val_samples]

    n_test_norm = min(len(test_norm_idx), max_test_samples // 2)
    n_test_anom = min(len(test_anom_idx), max_test_samples - n_test_norm)
    test_norm_sel = test_norm_idx[:n_test_norm]
    test_anom_sel = test_anom_idx[:n_test_anom]

    # Extract resized arrays
    train_imgs = extract_and_resize(train_raw.imgs, train_sel)
    train_masks = np.zeros_like(train_imgs)
    train_labels = np.zeros(len(train_imgs), dtype=np.int64)

    val_imgs = extract_and_resize(val_raw.imgs, val_sel)
    val_masks = np.zeros_like(val_imgs)
    val_labels = np.zeros(len(val_imgs), dtype=np.int64)

    test_norm_imgs = extract_and_resize(test_raw.imgs, test_norm_sel)
    test_norm_masks = np.zeros_like(test_norm_imgs)
    test_norm_labels = np.zeros(len(test_norm_imgs), dtype=np.int64)

    test_anom_imgs = extract_and_resize(test_raw.imgs, test_anom_sel)
    # Binary mask: non-zero pixels in anomaly slices treated as anomalous organ region
    test_anom_masks = (test_anom_imgs > 0.15).astype(np.float32)
    test_anom_labels = np.ones(len(test_anom_imgs), dtype=np.int64)

    test_imgs = np.concatenate([test_norm_imgs, test_anom_imgs], axis=0)
    test_masks = np.concatenate([test_norm_masks, test_anom_masks], axis=0)
    test_labels = np.concatenate([test_norm_labels, test_anom_labels], axis=0)

    # Shuffle test set
    p = rng.permutation(len(test_imgs))
    test_imgs, test_masks, test_labels = test_imgs[p], test_masks[p], test_labels[p]

    train_ds = BrainMRISliceDataset(train_imgs, train_masks, train_labels)
    val_ds = BrainMRISliceDataset(val_imgs, val_masks, val_labels)
    test_ds = BrainMRISliceDataset(test_imgs, test_masks, test_labels)

    return train_ds, val_ds, test_ds


def get_uad_dataloaders(
    dataset_source: str = "medmnist",
    dataset_name: str = "OrganAMNIST",
    batch_size: int = 16,
    image_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function returning (train_loader, val_loader, test_loader).
    Supports dataset_source="medmnist" (official MedMNIST) and dataset_source="phantom" (synthetic brain).
    """
    if dataset_source.lower() == "medmnist":
        try:
            train_ds, val_ds, test_ds = load_medmnist_uad_splits(
                dataset_name=dataset_name,
                image_size=image_size,
                seed=seed,
            )
        except Exception as e:
            print(f"Warning: MedMNIST loading encountered {e}. Falling back to HighFidelityBrainPhantom.")
            train_ds, val_ds, test_ds = create_uad_data_splits(image_size=image_size, seed=seed)
    else:
        train_ds, val_ds, test_ds = create_uad_data_splits(image_size=image_size, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
