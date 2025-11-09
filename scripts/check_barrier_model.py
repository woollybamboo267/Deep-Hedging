"""
Quick check to verify barrier option model is installed and working.

Usage:
    python scripts/check_barrier_model.py
    python scripts/check_barrier_model.py --model-path models/barrier/best_finetuned_up-and-in_call.1.pth
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_file_exists(model_path):
    """Check if model file exists."""
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"[✓] Barrier model found: {model_path}")
        print(f"    Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"[✗] Barrier model NOT found: {model_path}")
        print(f"\n[INFO] Download the model with:")
        print(f"       python scripts/download_models.py")
        return False


def check_model_loads(model_path):
    """Check if model can be loaded."""
    import torch
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"[✓] Model loads successfully")
        
        # Check required keys
        required_keys = ['model', 'mean', 'std']
        missing = [k for k in required_keys if k not in checkpoint]
        
        if missing:
            print(f"[!] Warning: Missing keys in checkpoint: {missing}")
        else:
            print(f"[✓] All required keys present: {required_keys}")
        
        # Check test MAE if available
        if 'test_mae' in checkpoint:
            print(f"[INFO] Test MAE: {checkpoint['test_mae']:.6f}")
        
        return True
    except Exception as e:
        print(f"[✗] Model load failed: {e}")
        return False


def check_model_inference(model_path):
    """Check if model can perform inference."""
    import torch
    from src.option_greek.barrier import BarrierOption
    
    try:
        # Create barrier option instance
        barrier = BarrierOption(
            model_path=model_path,
            barrier_level=120.0,
            barrier_type="up-and-in",
            option_type="call",
            r_annual=0.04,
            device="cpu"
        )
        print(f"[✓] BarrierOption instance created")
        
        # Test pricing
        S = torch.tensor([100.0], dtype=torch.float32)
        price = barrier.price(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        
        if torch.isfinite(price).all() and (price >= 0).all():
            print(f"[✓] Test pricing works: price={price.item():.4f}")
        else:
            print(f"[!] Warning: Price is invalid: {price.item()}")
            return False
        
        # Test Greeks
        delta = barrier.delta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        gamma = barrier.gamma(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        vega = barrier.vega(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        theta = barrier.theta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        
        if all(torch.isfinite(g).all() for g in [delta, gamma, vega, theta]):
            print(f"[✓] Greeks computation works")
            print(f"    Delta: {delta.item():.4f}")
            print(f"    Gamma: {gamma.item():.6f}")
            print(f"    Vega:  {vega.item():.4f}")
            print(f"    Theta: {theta.item():.4f}")
        else:
            print(f"[!] Warning: Some Greeks are invalid")
            return False
        
        return True
    
    except Exception as e:
        print(f"[✗] Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Check barrier option model setup")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/barrier/best_finetuned_up-and-in_call.1.pth",
        help="Path to barrier model"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Barrier Option Model Check")
    print("=" * 70)
    print()
    
    # Run checks
    checks_passed = 0
    total_checks = 3
    
    if check_file_exists(args.model_path):
        checks_passed += 1
        print()
        
        if check_model_loads(args.model_path):
            checks_passed += 1
            print()
            
            if check_model_inference(args.model_path):
                checks_passed += 1
    
    print()
    print("=" * 70)
    
    if checks_passed == total_checks:
        print(f"[SUCCESS] All checks passed ({checks_passed}/{total_checks})")
        print()
        print("You can now use barrier options:")
        print("  python train.py --config cfgs/config_barrier_2inst.yaml")
    else:
        print(f"[FAILURE] Some checks failed ({checks_passed}/{total_checks})")
        print()
        print("Please run the download script:")
        print("  python scripts/download_models.py")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
