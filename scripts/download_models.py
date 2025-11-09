"""
Download pre-trained barrier option models from GitHub releases.

The barrier option neural network model is ~107MB and stored in GitHub releases.
This script downloads it to the correct location.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir models/barrier
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_barrier_model(output_dir="models/barrier", force=False):
    """
    Download the barrier option model from GitHub releases.
    
    Args:
        output_dir: Directory to save the model
        force: If True, re-download even if file exists
    """
    # GitHub release URL (update this with your actual release URL)
    MODEL_URL = "https://github.com/woollybamboo267/Deep-Hedging/releases/download/%23surrogate/best_finetuned_up-and-in_call.1.pth"
    MODEL_FILENAME = "best_finetuned_up-and-in_call.1.pth"
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / MODEL_FILENAME
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"[INFO] Model already exists at {output_path}")
        print(f"[INFO] Size: {output_path.stat().st_size / 1e6:.1f} MB")
        print(f"[INFO] Use --force to re-download")
        return str(output_path)
    
    print(f"[INFO] Downloading barrier option model (~107 MB)...")
    print(f"[INFO] URL: {MODEL_URL}")
    print(f"[INFO] Destination: {output_path}")
    
    try:
        download_url(MODEL_URL, str(output_path))
        print(f"\n[SUCCESS] Model downloaded successfully!")
        print(f"[INFO] Size: {output_path.stat().st_size / 1e6:.1f} MB")
        return str(output_path)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"\n[INFO] Please download manually from:")
        print(f"       {MODEL_URL}")
        print(f"       and save to: {output_path}")
        sys.exit(1)


def verify_model(model_path):
    """Verify the downloaded model can be loaded."""
    import torch
    
    print(f"\n[INFO] Verifying model...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        required_keys = ['model', 'mean', 'std']
        for key in required_keys:
            if key not in checkpoint:
                print(f"[WARNING] Missing key in checkpoint: {key}")
        
        print(f"[SUCCESS] Model verification passed!")
        print(f"[INFO] Test MAE: {checkpoint.get('test_mae', 'N/A')}")
        
    except Exception as e:
        print(f"[ERROR] Model verification failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download barrier option models")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/barrier",
        help="Directory to save models (default: models/barrier)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip model verification after download"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Deep Hedging - Barrier Option Model Download")
    print("=" * 70)
    
    model_path = download_barrier_model(args.model_dir, args.force)
    
    if not args.no_verify:
        verify_model(model_path)
    
    print("\n" + "=" * 70)
    print("[INFO] Setup complete! You can now use barrier options:")
    print(f"       python train.py --config cfgs/config_barrier_2inst.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
