"""
Standalone script to precompute Heston-Nandi coefficients for all maturities.

This script precomputes coefficients for maturities from 1 to 504 days and
saves them to disk for reuse across training runs.

Usage:
    python precompute_all_maturities.py --output-dir precomputed_cache
    python precompute_all_maturities.py --min-maturity 1 --max-maturity 504
"""

import os
import sys
import argparse
import torch
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.option_greek.precompute import HestonNandiPrecomputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_and_save_all_maturities(
    garch_params: Dict[str, float],
    r_annual: float,
    min_maturity: int = 1,
    max_maturity: int = 504,
    output_dir: str = "precomputed_cache",
    N_quad: int = 256,
    u_max: float = 500.0,
    device: str = "cpu"
) -> None:
    """
    Precompute coefficients for all maturities and save to disk.
    
    Args:
        garch_params: GARCH model parameters
        r_annual: Annual risk-free rate
        min_maturity: Minimum maturity in days
        max_maturity: Maximum maturity in days
        output_dir: Directory to save precomputed data
        N_quad: Number of quadrature points
        u_max: Maximum integration range
        device: Computation device ('cpu' or 'cuda')
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("HESTON-NANDI PRECOMPUTATION")
    logger.info("="*70)
    logger.info(f"Maturity range: {min_maturity} to {max_maturity} days")
    logger.info(f"Total maturities: {max_maturity - min_maturity + 1}")
    logger.info(f"N_quad: {N_quad}")
    logger.info(f"u_max: {u_max}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)
    
    # Initialize precomputer
    r_daily = r_annual / 252.0
    precomputer = HestonNandiPrecomputer(
        garch_params=garch_params,
        r_daily=r_daily,
        N_quad=N_quad,
        u_max=u_max,
        device=device
    )
    
    logger.info("\nGARCH Parameters:")
    for key, value in garch_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  r_annual: {r_annual}")
    logger.info(f"  r_daily: {r_daily}")
    
    # Precompute for all maturities
    logger.info("\n" + "="*70)
    logger.info("Starting precomputation...")
    logger.info("="*70 + "\n")
    
    for N in range(min_maturity, max_maturity + 1):
        try:
            # Compute coefficients
            logger.info(f"[{N}/{max_maturity}] Computing maturity N={N} days...")
            
            precomputed_data = precomputer.precompute_coefficients(N)
            
            # Save to disk
            filename = output_path / f"precomputed_N{N:04d}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(precomputed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Log file size
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            logger.info(f"  ✓ Saved to {filename} ({file_size:.2f} MB)")
            
            # Progress indicator every 50 maturities
            if N % 50 == 0:
                progress = (N - min_maturity + 1) / (max_maturity - min_maturity + 1) * 100
                logger.info(f"\n  Progress: {progress:.1f}% complete\n")
        
        except Exception as e:
            logger.error(f"  ✗ Failed for N={N}: {e}")
            raise
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PRECOMPUTATION COMPLETE")
    logger.info("="*70)
    
    total_files = len(list(output_path.glob("precomputed_N*.pkl")))
    total_size = sum(f.stat().st_size for f in output_path.glob("precomputed_N*.pkl"))
    total_size_mb = total_size / (1024 * 1024)
    
    logger.info(f"Total files created: {total_files}")
    logger.info(f"Total size: {total_size_mb:.2f} MB")
    logger.info(f"Average file size: {total_size_mb/total_files:.2f} MB")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)


def load_precomputed_data(maturity: int, cache_dir: str = "precomputed_cache") -> Dict[str, Any]:
    """
    Load precomputed data for a specific maturity from disk.
    
    Args:
        maturity: Maturity in days
        cache_dir: Directory containing precomputed data
        
    Returns:
        Dictionary containing precomputed coefficients and metadata
    """
    filename = Path(cache_dir) / f"precomputed_N{maturity:04d}.pkl"
    
    if not filename.exists():
        raise FileNotFoundError(f"Precomputed data not found: {filename}")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded precomputed data for N={maturity} from {filename}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Heston-Nandi coefficients for all maturities"
    )
    parser.add_argument(
        "--min-maturity",
        type=int,
        default=1,
        help="Minimum maturity in days (default: 1)"
    )
    parser.add_argument(
        "--max-maturity",
        type=int,
        default=504,
        help="Maximum maturity in days (default: 504)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="precomputed_cache",
        help="Output directory for precomputed data (default: precomputed_cache)"
    )
    parser.add_argument(
        "--N-quad",
        type=int,
        default=256,
        help="Number of quadrature points (default: 256)"
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=500.0,
        help="Maximum integration range (default: 500.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    elif torch.cuda.is_available() and args.device == "cpu":
        logger.info("CUDA is available but CPU was selected")
    
    # Standard GARCH parameters (from your config)
    garch_params = {
        "omega": 1.593749e-07,
        "alpha": 2.308475e-06,
        "beta": 0.689984,
        "gamma": 342.870019,
        "lambda": 0.420499,
        "sigma0": 0.127037
    }
    
    r_annual = 0.04
    
    # Run precomputation
    precompute_and_save_all_maturities(
        garch_params=garch_params,
        r_annual=r_annual,
        min_maturity=args.min_maturity,
        max_maturity=args.max_maturity,
        output_dir=args.output_dir,
        N_quad=args.N_quad,
        u_max=args.u_max,
        device=args.device
    )


if __name__ == "__main__":
    main()
