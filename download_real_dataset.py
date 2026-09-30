import urllib.request
import zipfile
import io
import os
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data") / "brain_tumor_dataset"
ZIP_URL = "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection/resolve/main/Brain%20MRI%20Images%20for%20Brain%20Tumor%20Detection.zip"

print(f"Downloading real clinical Brain MRI dataset from Hugging Face...")
req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    zip_bytes = resp.read()

print(f"Downloaded {len(zip_bytes) / (1024*1024):.2f} MB. Extracting to {DATA_DIR}...")
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    zf.extractall("data")

print("Extraction complete!")
no_dir = DATA_DIR / "no"
yes_dir = DATA_DIR / "yes"

no_files = list(no_dir.glob("*.*"))
yes_files = list(yes_dir.glob("*.*"))
print(f"Extracted {len(no_files)} healthy (no) scans, {len(yes_files)} pathological (yes) scans.")

print("\nSample Healthy Scans:")
for p in no_files[:5]:
    with Image.open(p) as img:
        print(f"  {p.name}: {img.size[0]}x{img.size[1]} (W x H), Aspect Ratio: {img.size[0]/img.size[1]:.2f}, Mode: {img.mode}")

print("\nSample Tumor Scans:")
for p in yes_files[:5]:
    with Image.open(p) as img:
        print(f"  {p.name}: {img.size[0]}x{img.size[1]} (W x H), Aspect Ratio: {img.size[0]/img.size[1]:.2f}, Mode: {img.mode}")
