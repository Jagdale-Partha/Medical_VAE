"""
prepare_ixi_t1.py - Prepares IXI T1 Brain MRI Dataset for ceVAE+ Training and Evaluation

Extracts preprocessed 2D T1 brain MRI slices (fsaverage registered, white-matter normalized)
into standardized UAD splits:
- data/IXI-T1/train/ (healthy normative scans, y=0)
- data/IXI-T1/val/   (healthy normative scans, y=0)
- data/IXI-T1/test_healthy/ (healthy normative scans for UAD evaluation)
"""

import os
import zipfile
import io
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

def prepare_dataset(
    train_zip_path: str,
    valid_zip_path: str,
    output_dir: str = "data/IXI-T1",
    num_train: int = 5000,
    num_val: int = 800,
    num_test_healthy: int = 500,
    target_size: int = 128,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    
    out_path = Path(output_dir)
    train_dir = out_path / "train"
    val_dir = out_path / "val"
    test_h_dir = out_path / "test_healthy"
    
    for d in [train_dir, val_dir, test_h_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Reading valid zip: {valid_zip_path}...")
    with zipfile.ZipFile(valid_zip_path, "r") as z_val:
        val_members = [m for m in z_val.namelist() if m.endswith(".jpeg") and not m.startswith("__")]
        random.shuffle(val_members)
        
        # Take validation and test_healthy from valid_zip
        val_slice_names = val_members[:num_val]
        test_slice_names = val_members[num_val : num_val + num_test_healthy]
        
        print(f"Extracting {len(val_slice_names)} validation slices to {val_dir} (resized to {target_size}x{target_size})...")
        for i, name in enumerate(val_slice_names):
            data = z_val.read(name)
            img = Image.open(io.BytesIO(data)).convert("L")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            dest_file = val_dir / f"ixi_t1_val_{i:05d}.png"
            img.save(dest_file, "PNG")
            
        print(f"Extracting {len(test_slice_names)} test_healthy slices to {test_h_dir}...")
        for i, name in enumerate(test_slice_names):
            data = z_val.read(name)
            img = Image.open(io.BytesIO(data)).convert("L")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            dest_file = test_h_dir / f"ixi_t1_test_h_{i:05d}.png"
            img.save(dest_file, "PNG")

    print(f"Reading train zip: {train_zip_path}...")
    with zipfile.ZipFile(train_zip_path, "r") as z_train:
        train_members = [m for m in z_train.namelist() if m.endswith(".jpeg") and not m.startswith("__")]
        random.shuffle(train_members)
        
        train_slice_names = train_members[:num_train]
        print(f"Extracting {len(train_slice_names)} train slices to {train_dir} (resized to {target_size}x{target_size})...")
        for i, name in enumerate(train_slice_names):
            data = z_train.read(name)
            img = Image.open(io.BytesIO(data)).convert("L")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            dest_file = train_dir / f"ixi_t1_train_{i:05d}.png"
            img.save(dest_file, "PNG")
            if (i + 1) % 1000 == 0 or (i + 1) == len(train_slice_names):
                print(f"  Processed {i + 1}/{len(train_slice_names)} training slices...")

    print("IXI T1 Dataset preparation complete!")
    print(f"Train slices: {len(list(train_dir.glob('*.png')))}")
    print(f"Validation slices: {len(list(val_dir.glob('*.png')))}")
    print(f"Test healthy slices: {len(list(test_h_dir.glob('*.png')))}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare IXI T1 dataset")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training slices")
    parser.add_argument("--num_val", type=int, default=800, help="Number of validation slices")
    parser.add_argument("--num_test_healthy", type=int, default=500, help="Number of test healthy slices")
    parser.add_argument("--size", type=int, default=128, help="Target image size (default: 128)")
    args = parser.parse_args()

    train_zip = r"C:\Users\jagda\.cache\huggingface\hub\datasets--iamkzntsv--IXI2D\snapshots\292eb48a52af9914c36642fed7cd01241a698268\data\train.zip"
    valid_zip = r"C:\Users\jagda\.cache\huggingface\hub\datasets--iamkzntsv--IXI2D\snapshots\292eb48a52af9914c36642fed7cd01241a698268\data\valid.zip"

    prepare_dataset(
        train_zip_path=train_zip,
        valid_zip_path=valid_zip,
        output_dir="data/IXI-T1",
        num_train=args.num_train,
        num_val=args.num_val,
        num_test_healthy=args.num_test_healthy,
        target_size=args.size,
    )
