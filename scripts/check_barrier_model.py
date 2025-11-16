"""
Quick check to verify surrogate models are installed and working.

Usage:
    python scripts/check_models.py                  # Check all models
    python scripts/check_models.py --model american # Check only American
    python scripts/check_models.py --model barrier  # Check only Barrier
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Model configurations
MODELS = {
    "american": {
        "path": "models/american/discriminative_v5_american_put.pth",
        "expected_size_mb": 26.9,
        "option_class": "AmericanOption",
        "config": "cfgs/config_american_2inst.yaml"
    },
    "barrier": {
        "path": "models/barrier/best_finetuned_up-and-in_call.1.pth",
        "expected_size_mb": 107.0,
        "option_class": "BarrierOption",
        "config": "cfgs/config_barrier_2inst.yaml"
    }
}


def check_file_exists(model_name):
    """Check if model file exists."""
    config = MODELS[model_name]
    model_path = config["path"]
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"[✓] {model_name.capitalize()} model found: {model_path}")
        print(f"    Size: {size_mb:.1f} MB (expected: ~{config['expected_size_mb']:.1f} MB)")
        
        # Check if size is reasonable
        if abs(size_mb - config['expected_size_mb']) / config['expected_size_mb'] > 0.1:
            print(f"    [!] Warning: Size differs significantly from expected")
        
        return True
    else:
        print(f"[✗] {model_name.capitalize()} model NOT found: {model_path}")
        return False


def check_model_loads(model_name):
    """Check if model can be loaded."""
    import torch
    
    config = MODELS[model_name]
    model_path = config["path"]
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"[✓] {model_name.capitalize()} model loads successfully")
        
        # Check for common keys
        common_keys = ['model', 'mean', 'std']
        found_keys = [k for k in common_keys if k in checkpoint]
        missing_keys = [k for k in common_keys if k not in checkpoint]
        
        if found_keys:
            print(f"[✓] Found expected keys: {', '.join(found_keys)}")
        
        if missing_keys:
            print(f"[!] Warning: Missing keys: {', '.join(missing_keys)}")
        
        # Check test MAE if available
        if 'test_mae' in checkpoint:
            print(f"[INFO] Test MAE: {checkpoint['test_mae']:.6f}")
        
        return True
    except Exception as e:
        print(f"[✗] {model_name.capitalize()} model load failed: {e}")
        return False


def check_barrier_inference():
    """Check if barrier model can perform inference."""
    import torch
    from src.option_greek.barrier import BarrierOption
    
    try:
        barrier = BarrierOption(
            model_path=MODELS["barrier"]["path"],
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
            print(f"[✓] Barrier pricing works: price={price.item():.4f}")
        else:
            print(f"[!] Warning: Price is invalid: {price.item()}")
            return False
        
        # Test Greeks
        delta = barrier.delta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        gamma = barrier.gamma(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        vega = barrier.vega(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        theta = barrier.theta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        
        if all(torch.isfinite(g).all() for g in [delta, gamma, vega, theta]):
            print(f"[✓] Barrier Greeks computation works")
            print(f"    Delta: {delta.item():.4f}, Gamma: {gamma.item():.6f}")
            print(f"    Vega:  {vega.item():.4f}, Theta: {theta.item():.4f}")
        else:
            print(f"[!] Warning: Some Greeks are invalid")
            return False
        
        return True
    
    except Exception as e:
        print(f"[✗] Barrier inference test failed: {e}")
        return False


def check_american_inference():
    """Check if american model can perform inference."""
    import torch
    from src.option_greek.american import AmericanOption
    
    try:
        american = AmericanOption(
            model_path=MODELS["american"]["path"],
            option_type="put",
            r_annual=0.04,
            device="cpu"
        )
        print(f"[✓] AmericanOption instance created")
        
        # Test pricing
        S = torch.tensor([100.0], dtype=torch.float32)
        price = american.price(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        
        if torch.isfinite(price).all() and (price >= 0).all():
            print(f"[✓] American pricing works: price={price.item():.4f}")
        else:
            print(f"[!] Warning: Price is invalid: {price.item()}")
            return False
        
        # Test Greeks
        delta = american.delta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        gamma = american.gamma(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        vega = american.vega(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        theta = american.theta(S=S, K=100.0, step_idx=0, N=252, h0=5.14e-7)
        
        if all(torch.isfinite(g).all() for g in [delta, gamma, vega, theta]):
            print(f"[✓] American Greeks computation works")
            print(f"    Delta: {delta.item():.4f}, Gamma: {gamma.item():.6f}")
            print(f"    Vega:  {vega.item():.4f}, Theta: {theta.item():.4f}")
        else:
            print(f"[!] Warning: Some Greeks are invalid")
            return False
        
        return True
    
    except Exception as e:
        print(f"[✗] American inference test failed: {e}")
        return False


def check_model(model_name):
    """Run all checks for a specific model."""
    print(f"\nChecking {model_name.upper()} model")
    print("-" * 70)
    
    checks_passed = 0
    total_checks = 3
    
    # Check 1: File exists
    if not check_file_exists(model_name):
        print(f"\n[INFO] Download the {model_name} model with:")
        print(f"       python scripts/download_models.py --model {model_name}")
        return 0, total_checks
    
    checks_passed += 1
    print()
    
    # Check 2: Model loads
    if not check_model_loads(model_name):
        return checks_passed, total_checks
    
    checks_passed += 1
    print()
    
    # Check 3: Inference works
    try:
        if model_name == "barrier":
            inference_ok = check_barrier_inference()
        elif model_name == "american":
            inference_ok = check_american_inference()
        else:
            print(f"[!] No inference test available for {model_name}")
            inference_ok = True
        
        if inference_ok:
            checks_passed += 1
    except ImportError as e:
        print(f"[!] Cannot test inference: {e}")
        print(f"[INFO] Make sure the option classes are implemented")
    
    return checks_passed, total_checks


def main():
    parser = argparse.ArgumentParser(
        description="Check surrogate model setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_models.py                  # Check all models
  python scripts/check_models.py --model american # Check only American
  python scripts/check_models.py --model barrier  # Check only Barrier
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Specific model to check (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Surrogate Models Check")
    print("=" * 70)
    
    # Determine which models to check
    if args.model == "all":
        models_to_check = list(MODELS.keys())
    else:
        models_to_check = [args.model]
    
    # Check each model
    results = {}
    for model_name in models_to_check:
        passed, total = check_model(model_name)
        results[model_name] = (passed, total)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = True
    for model_name, (passed, total) in results.items():
        status = "✓ PASS" if passed == total else "✗ FAIL"
        print(f"{status} {model_name.capitalize()}: {passed}/{total} checks passed")
        if passed < total:
            all_passed = False
    
    print()
    
    if all_passed:
        print("[SUCCESS] All models are ready to use!")
        print()
        print("You can now train with:")
        for model_name in models_to_check:
            if model_name in MODELS:
                print(f"  python train.py --config {MODELS[model_name]['config']}")
    else:
        print("[FAILURE] Some models are not ready")
        print()
        print("Download missing models with:")
        print("  python scripts/download_models.py")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
