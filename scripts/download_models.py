"""
Download pre-trained surrogate models from GitHub releases.

This script downloads American and Barrier option neural network models (~26-107MB each)
stored in GitHub releases to the correct locations.

Usage:
    python scripts/download_models.py                    # Download all models
    python scripts/download_models.py --model american   # Download only American
    python scripts/download_models.py --model barrier    # Download only Barrier
    python scripts/download_models.py --force            # Force re-download
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Model configurations
MODELS = {
    "american": {
        "url": "https://github.com/woollybamboo267/Deep-Hedging/releases/download/surrogate1/discriminative_v5_american_put.pth",
        "filename": "discriminative_v5_american_put.pth",
        "output_dir": "models/american",
        "size_mb": 26.9,
        "description": "American option surrogate model"
    },
    "barrier": {
        "url": "https://github.com/woollybamboo267/Deep-Hedging/releases/download/%23surrogate/best_finetuned_up-and-in_call.1.pth",
        "filename": "best_finetuned_up-and-in_call.1.pth",
        "output_dir": "models/barrier",
        "size_mb": 107.0,
        "description": "Barrier option surrogate model"
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_model(model_name, force=False):
    """
    Download a specific surrogate model from GitHub releases.
    
    Args:
        model_name: Name of the model ('american' or 'barrier')
        force: If True, re-download even if file exists
        
    Returns:
        Path to downloaded model or None if failed
    """
    if model_name not in MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"[INFO] Available models: {', '.join(MODELS.keys())}")
        return None
    
    config = MODELS[model_name]
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / config["filename"]
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"[INFO] {config['description']} already exists at {output_path}")
        print(f"[INFO] Size: {output_path.stat().st_size / 1e6:.1f} MB")
        print(f"[INFO] Use --force to re-download")
        return str(output_path)
    
    print(f"\n[INFO] Downloading {config['description']} (~{config['size_mb']:.1f} MB)...")
    print(f"[INFO] URL: {config['url']}")
    print(f"[INFO] Destination: {output_path}")
    
    try:
        download_url(config["url"], str(output_path))
        print(f"\n[SUCCESS] {model_name.capitalize()} model downloaded successfully!")
        print(f"[INFO] Size: {output_path.stat().st_size / 1e6:.1f} MB")
        return str(output_path)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"\n[INFO] Please download manually from:")
        print(f"       {config['url']}")
        print(f"       and save to: {output_path}")
        return None


def verify_model(model_path, model_name):
    """Verify the downloaded model can be loaded."""
    import torch
    
    print(f"\n[INFO] Verifying {model_name} model...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check for common keys (models may have different structures)
        common_keys = ['model', 'mean', 'std']
        found_keys = [k for k in common_keys if k in checkpoint]
        
        if found_keys:
            print(f"[SUCCESS] Model verification passed!")
            print(f"[INFO] Found keys: {', '.join(found_keys)}")
        else:
            print(f"[WARNING] Checkpoint structure may differ from expected")
            print(f"[INFO] Available keys: {list(checkpoint.keys())[:5]}...")
        
        # Check test MAE if available
        if 'test_mae' in checkpoint:
            print(f"[INFO] Test MAE: {checkpoint.get('test_mae', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download surrogate option models from GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_models.py                    # Download all models
  python scripts/download_models.py --model american   # Download only American
  python scripts/download_models.py --model barrier    # Download only Barrier
  python scripts/download_models.py --force            # Force re-download all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Specific model to download (default: all)"
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
    print("Deep Hedging - Surrogate Model Download")
    print("=" * 70)
    
    # Determine which models to download
    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]
    
    print(f"[INFO] Models to download: {', '.join(models_to_download)}")
    
    # Download each model
    downloaded_models = []
    failed_models = []
    
    for model_name in models_to_download:
        model_path = download_model(model_name, args.force)
        
        if model_path:
            downloaded_models.append((model_name, model_path))
            
            if not args.no_verify:
                if not verify_model(model_path, model_name):
                    failed_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    if downloaded_models:
        print(f"\n[SUCCESS] Downloaded {len(downloaded_models)} model(s):")
        for model_name, model_path in downloaded_models:
            print(f"  ✓ {model_name}: {model_path}")
    
    if failed_models:
        print(f"\n[WARNING] Failed to download {len(failed_models)} model(s):")
        for model_name in failed_models:
            print(f"  ✗ {model_name}")
    
    if downloaded_models and not failed_models:
        print("\n" + "=" * 70)
        print("[INFO] Setup complete! You can now use the models:")
        if "american" in [m[0] for m in downloaded_models]:
            print(f"       python train.py --config cfgs/config_american_2inst.yaml")
        if "barrier" in [m[0] for m in downloaded_models]:
            print(f"       python train.py --config cfgs/config_barrier_2inst.yaml")
        print("=" * 70)
    
    if failed_models:
        sys.exit(1)


if __name__ == "__main__":
    main()
