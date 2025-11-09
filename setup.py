"""
One-command setup script for Deep Hedging.

Usage:
    python setup.py                    # Full setup with barrier models
    python setup.py --vanilla-only     # Skip barrier model download
    python setup.py --check-only       # Only verify installation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def check_python_version():
    """Check Python version."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"[✗] Python 3.8+ required, found {major}.{minor}")
        return False
    print(f"[✓] Python version: {major}.{minor}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'numba': 'Numba',
        'yaml': 'PyYAML',
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"[✓] {name} installed")
        except ImportError:
            print(f"[✗] {name} missing")
            missing.append(name)
    
    return missing


def install_dependencies(missing):
    """Install missing dependencies."""
    if not missing:
        return True
    
    print(f"\n[INFO] Installing missing packages: {', '.join(missing)}")
    
    # Map display names to pip package names
    pip_names = {
        'PyTorch': 'torch',
        'NumPy': 'numpy',
        'Numba': 'numba',
        'PyYAML': 'pyyaml',
        'tqdm': 'tqdm'
    }
    
    packages = [pip_names.get(name, name.lower()) for name in missing]
    
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install'] + packages
        )
        print(f"[✓] Packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"[✗] Installation failed")
        print(f"[INFO] Please install manually: pip install {' '.join(packages)}")
        return False


def check_project_structure():
    """Verify project structure."""
    required_dirs = [
        'src/agents',
        'src/option_greek',
        'src/simulation',
        'src/visualization',
        'cfgs',
        'scripts',
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
            print(f"[✗] Missing directory: {dir_path}")
        else:
            print(f"[✓] {dir_path}")
    
    if missing:
        print(f"\n[ERROR] Project structure incomplete")
        print(f"[INFO] Are you in the project root directory?")
        return False
    
    return True


def setup_barrier_models():
    """Download barrier option models."""
    print_header("Setting up barrier option models")
    
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/download_models.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            print(f"\n[!] Barrier model download failed")
            print(f"[INFO] You can download manually later:")
            print(f"       python scripts/download_models.py")
            return False
    
    except Exception as e:
        print(f"[!] Could not download barrier models: {e}")
        print(f"[INFO] You can download manually later:")
        print(f"       python scripts/download_models.py")
        return False


def verify_barrier_setup():
    """Verify barrier option setup."""
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/check_barrier_model.py'],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        return result.returncode == 0
    
    except Exception as e:
        print(f"[!] Could not verify barrier setup: {e}")
        return False


def create_models_directory():
    """Create models directory structure."""
    dirs = [
        'models/barrier',
        'models/uniform',
        'models/non-uniform',
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[✓] Created {dir_path}")


def print_usage_instructions(barrier_enabled):
    """Print usage instructions."""
    print_header("Setup Complete!")
    
    print("Quick Start:")
    print()
    
    if barrier_enabled:
        print("# Barrier option hedging")
        print("python train.py --config cfgs/config_barrier_2inst.yaml")
        print()
    
    print("# Vanilla option hedging")
    print("python train.py --config cfgs/config_vanilla_2inst.yaml")
    print()
    print("# Custom configuration")
    print("python train.py --config cfgs/my_config.yaml")
    print()
    print("# Inference only")
    print("python train.py --config PATH --load-model PATH --inference-only")
    print()
    
    print("Documentation:")
    print("  - README.md           - Project overview")
    print("  - BARRIER_SETUP.md    - Barrier option details")
    print()
    
    print("Troubleshooting:")
    print("  - Check barrier setup: python scripts/check_barrier_model.py")
    print("  - Re-run setup:        python setup.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="Setup Deep Hedging project")
    parser.add_argument(
        "--vanilla-only",
        action="store_true",
        help="Skip barrier model download (vanilla options only)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify installation, don't install anything"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Don't install missing dependencies"
    )
    
    args = parser.parse_args()
    
    print_header("Deep Hedging - Setup Script")
    
    # Check Python version
    print_header("Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print_header("Checking dependencies")
    missing = check_dependencies()
    
    if missing:
        if args.check_only or args.no_install:
            print(f"\n[INFO] Missing dependencies: {', '.join(missing)}")
            print(f"[INFO] Install with: pip install {' '.join(m.lower() for m in missing)}")
            if args.check_only:
                sys.exit(1)
        else:
            if not install_dependencies(missing):
                sys.exit(1)
    
    # Check project structure
    print_header("Checking project structure")
    if not check_project_structure():
        sys.exit(1)
    
    # Create directories
    print_header("Creating directory structure")
    if not args.check_only:
        create_models_directory()
    
    # Setup barrier models
    barrier_enabled = False
    if not args.vanilla_only and not args.check_only:
        if setup_barrier_models():
            barrier_enabled = True
            print_header("Verifying barrier setup")
            verify_barrier_setup()
    
    # Print usage
    print_usage_instructions(barrier_enabled)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
