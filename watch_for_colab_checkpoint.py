"""
watch_for_colab_checkpoint.py - Automated Checkpoint Receiver & Showcase Runner
Monitors the user's local Downloads folder and checkpoints/ directory for best_cevae_model.pt
downloaded from Google Colab. Once detected, it:
1. Verifies checkpoint integrity and extracts epoch/val_loss metadata.
2. Moves/saves it to checkpoints/best_cevae_model.pt.
3. Automatically runs showcase_samples.py to generate 5 healthy and 5 unhealthy showcase galleries.
"""

import sys
import time
import shutil
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_CHK = CHECKPOINTS_DIR / "best_cevae_model.pt"

DOWNLOADS_DIR = Path.home() / "Downloads"


def verify_checkpoint(path: Path) -> dict:
    # Give browser a second to finish file write if needed
    for attempt in range(5):
        try:
            chk = torch.load(path, map_location="cpu", weights_only=False)
            return {
                "valid": True,
                "epoch": chk.get("epoch", "unknown"),
                "val_loss": chk.get("val_loss", None),
                "size_mb": path.stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            time.sleep(2)
    return {"valid": False, "error": str(e)}


def main(timeout_seconds: int = 3600, poll_interval: int = 5):
    print("=============================================================", flush=True)
    print("📡 ceVAE+ Checkpoint Receiver & Showcase Runner Active", flush=True)
    print(f"Monitoring Downloads folder: {DOWNLOADS_DIR}", flush=True)
    print(f"Target destination:        {TARGET_CHK}", flush=True)
    print(f"Listening for 50-epoch checkpoint from Colab (Timeout: {timeout_seconds // 60} mins)...", flush=True)
    print("=============================================================\n", flush=True)

    initial_mtime = TARGET_CHK.stat().st_mtime if TARGET_CHK.is_file() else 0
    start_time = time.time()
    last_ping = start_time

    while (time.time() - start_time) < timeout_seconds:
        # Check Downloads folder for any best_cevae_model*.pt
        dl_candidates = list(DOWNLOADS_DIR.glob("best_cevae_model*.pt"))
        if dl_candidates:
            # Sort newest first
            dl_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidate = dl_candidates[0]
            # Ignore files that are still being written by Chrome (.crdownload)
            if not any(DOWNLOADS_DIR.glob("best_cevae_model*.crdownload")):
                if candidate.stat().st_mtime >= (start_time - 30):
                    print(f"\n[FOUND] Detected fresh Colab checkpoint in Downloads: {candidate.name}", flush=True)
                    info = verify_checkpoint(candidate)
                    if info["valid"]:
                        print(f"  Integrity Check: PASSED ({info['size_mb']:.2f} MB)", flush=True)
                        print(f"  Trained Epoch:   {info['epoch']}", flush=True)
                        print(f"  Validation Loss: {info['val_loss']}", flush=True)
                        print(f"  Moving to {TARGET_CHK}...", flush=True)
                        shutil.copy2(candidate, TARGET_CHK)
                        print(f"✓ Checkpoint installed successfully at {TARGET_CHK}", flush=True)
                        break

        # Check if checkpoints/best_cevae_model.pt was updated directly
        if TARGET_CHK.is_file() and TARGET_CHK.stat().st_mtime > (initial_mtime + 2):
            info = verify_checkpoint(TARGET_CHK)
            # Make sure it's a freshly updated checkpoint (not the initial 1-epoch test)
            if info["valid"] and (info.get("epoch") != 0 or TARGET_CHK.stat().st_mtime > start_time):
                print(f"\n[FOUND] Detected updated checkpoint at {TARGET_CHK}", flush=True)
                print(f"  Trained Epoch:   {info['epoch']}", flush=True)
                print(f"  Validation Loss: {info['val_loss']}", flush=True)
                break

        # Heartbeat every 60 seconds
        if time.time() - last_ping >= 60:
            elapsed_min = int((time.time() - start_time) // 60)
            print(f"  [Waiting...] {elapsed_min}m elapsed. Monitoring {DOWNLOADS_DIR}...", flush=True)
            last_ping = time.time()

        time.sleep(poll_interval)
    else:
        print("Notice: Timeout reached waiting for new checkpoint.", flush=True)
        return

    # Trigger showcase generation
    print("\n🚀 Triggering 5 Healthy + 5 Unhealthy Diagnostic Showcase on fresh weights...", flush=True)
    from showcase_samples import run_showcase
    run_showcase(checkpoint_path=str(TARGET_CHK), output_dir=str(PROJECT_ROOT / "results"))
    print("\n=============================================================", flush=True)
    print("✓ Full pipeline finished! All showcase galleries updated.", flush=True)
    print("=============================================================", flush=True)


if __name__ == "__main__":
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
    main(timeout_seconds=timeout)
