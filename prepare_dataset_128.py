"""
prepare_dataset_128.py - Standardized 128x128 Medical Brain MRI Dataset Preparation
Organizes downsampled normative IXI T1 scans and clinical pathological scans into:
  data/dataset_128/
    ├── train/           (Healthy normative IXI T1 128x128)
    ├── val/             (Healthy normative IXI T1 128x128)
    ├── test_healthy/    (Healthy normative IXI T1 128x128)
    ├── test_patho/      (Clinical brain tumor scans letterboxed & downsampled to 128x128)
    ├── train_128.npy
    ├── val_128.npy
    ├── test_healthy_128.npy
    └── test_patho_128.npy
"""

import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np

def prepare_128_dataset(
    src_ixi_dir: str = "data/IXI-T1",
    src_tumor_dir: str = "data/brain_tumor_dataset/yes",
    dest_dir: str = "data/dataset_128",
    num_train: int = 4000,
    num_val: int = 600,
    num_test_healthy: int = 400,
    num_test_patho: int = 60,
    target_size: int = 128,
):
    dest = Path(dest_dir)
    train_dir = dest / "train"
    val_dir = dest / "val"
    test_h_dir = dest / "test_healthy"
    test_p_dir = dest / "test_patho"

    for d in [train_dir, val_dir, test_h_dir, test_p_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== Preparing Standardized 128x128 Brain MRI Dataset ===")

    # 1. Healthy Train
    train_files = sorted(glob.glob(f"{src_ixi_dir}/train/*.png"))[:num_train]
    print(f"[1/4] Copying & verifying {len(train_files)} training slices...")
    train_arr = np.zeros((len(train_files), target_size, target_size), dtype=np.uint8)
    for i, f in enumerate(train_files):
        img = Image.open(f).convert("L")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img.save(train_dir / f"train_{i:05d}.png")
        train_arr[i] = np.array(img)
    np.save(dest / "train_128.npy", train_arr)
    print(f"  Saved train_128.npy: shape {train_arr.shape}")

    # 2. Healthy Val
    val_files = sorted(glob.glob(f"{src_ixi_dir}/val/*.png"))[:num_val]
    print(f"[2/4] Copying & verifying {len(val_files)} validation slices...")
    val_arr = np.zeros((len(val_files), target_size, target_size), dtype=np.uint8)
    for i, f in enumerate(val_files):
        img = Image.open(f).convert("L")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img.save(val_dir / f"val_{i:05d}.png")
        val_arr[i] = np.array(img)
    np.save(dest / "val_128.npy", val_arr)
    print(f"  Saved val_128.npy: shape {val_arr.shape}")

    # 3. Healthy Test
    test_h_files = sorted(glob.glob(f"{src_ixi_dir}/test_healthy/*.png"))[:num_test_healthy]
    print(f"[3/4] Copying & verifying {len(test_h_files)} test healthy slices...")
    test_h_arr = np.zeros((len(test_h_files), target_size, target_size), dtype=np.uint8)
    for i, f in enumerate(test_h_files):
        img = Image.open(f).convert("L")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img.save(test_h_dir / f"test_h_{i:05d}.png")
        test_h_arr[i] = np.array(img)
    np.save(dest / "test_healthy_128.npy", test_h_arr)
    print(f"  Saved test_healthy_128.npy: shape {test_h_arr.shape}")

    # 4. Pathological Test (Downsampled clinical tumor scans with aspect-ratio letterboxing)
    patho_candidates = sorted(glob.glob(f"{src_tumor_dir}/*"))
    valid_patho_imgs = []
    print(f"[4/4] Downsampling & letterboxing clinical tumor scans to 128x128...")
    for f in patho_candidates:
        try:
            with Image.open(f) as im:
                w, h = im.size
                aspect = min(w, h) / max(w, h)
                if aspect >= 0.70:  # axial cranial scans
                    im_gray = im.convert("L")
                    scale = target_size / max(w, h)
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    im_res = im_gray.resize((nw, nh), Image.Resampling.BILINEAR)

                    canvas = Image.new("L", (target_size, target_size), 0)
                    canvas.paste(im_res, ((target_size - nw) // 2, (target_size - nh) // 2))
                    valid_patho_imgs.append(canvas)
                    if len(valid_patho_imgs) >= num_test_patho:
                        break
        except Exception:
            continue

    patho_arr = np.zeros((len(valid_patho_imgs), target_size, target_size), dtype=np.uint8)
    for i, im in enumerate(valid_patho_imgs):
        im.save(test_p_dir / f"test_patho_{i:05d}.png")
        patho_arr[i] = np.array(im)
    np.save(dest / "test_patho_128.npy", patho_arr)
    print(f"  Saved test_patho_128.npy: shape {patho_arr.shape}")

    print("\nDataset 128x128 preparation successfully finished!")
    print(f"Directory: {dest.resolve()}")

if __name__ == "__main__":
    prepare_128_dataset()
