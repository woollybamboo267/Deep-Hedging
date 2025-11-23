import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
"""
Precomputation module for Heston-Nandi option pricing.

This module handles the precomputation of characteristic function coefficients
for the Heston-Nandi GARCH option pricing model. These coefficients are used
for efficient option pricing, delta, gamma, and vega calculations.
"""

logger = logging.getLogger(__name__)


class HestonNandiPrecomputer:
    """
    Precomputes and manages Heston-Nandi characteristic function coefficients.
    
    This class handles the precomputation of coefficients needed for the
    Heston-Nandi GARCH option pricing model, which are then used for
    efficient pricing and greek calculations.
    """
    
    def __init__(
        self,
        garch_params: Dict[str, float],
        r_daily: float,
        N_quad: int = 128,
        u_max: float = 100.0,
        device: str = "cpu"
    ):
        """
        Initialize the precomputer.
        
        Args:
            garch_params: Dictionary containing GARCH parameters
                - omega: GARCH omega parameter
                - alpha: GARCH alpha parameter
                - beta: GARCH beta parameter
                - gamma: GARCH gamma parameter
                - lambda: Risk premium parameter
                - sigma0: Initial volatility
            r_daily: Daily risk-free rate
            N_quad: Number of quadrature points for numerical integration
            u_max: Maximum integration range
            device: Torch device ('cpu' or 'cuda')
        """
        self.garch_params = garch_params
        self.r_daily = r_daily
        self.N_quad = N_quad
        self.u_max = u_max
        self.device = torch.device(device)
        
        # Extract GARCH parameters
        self.omega = float(garch_params["omega"])
        self.alpha = float(garch_params["alpha"])
        self.beta = float(garch_params["beta"])
        self.gamma = float(garch_params["gamma"])
        self.lambda_ = float(garch_params["lambda"])
        self.sigma0 = float(garch_params["sigma0"])
        
        logger.info(
            f"Initialized HestonNandiPrecomputer with N_quad={N_quad}, "
            f"u_max={u_max}, device={device}"
        )
    
    def precompute_coefficients(self, N: int) -> Dict[str, Any]:
        """
        Precompute characteristic function coefficients for a given maturity.
        
        This method performs the recursive calculation of the characteristic
        function coefficients that are used in the Heston-Nandi pricing formula.
        
        Args:
            N: Number of time steps (maturity in days)
            
        Returns:
            Dictionary containing:
                - coefficients: Tensor of shape [N+1, N_quad, 2, 3]
                - u_nodes: Quadrature nodes
                - w_nodes: Quadrature weights
                - N: Number of time steps
                - r_daily: Daily risk-free rate
                - device: Device string
        """
        logger.info(f"Precomputing coefficients for N={N} time steps...")
        
        # Setup Gauss-Legendre quadrature
        u_nodes, w_nodes = leggauss(self.N_quad)
        u_nodes = 0.5 * (u_nodes + 1) * self.u_max
        w_nodes = 0.5 * self.u_max * w_nodes
        
        u_nodes_t = torch.tensor(u_nodes, dtype=torch.float64, device=self.device)
        w_nodes_t = torch.tensor(w_nodes, dtype=torch.float64, device=self.device)
        
        # Convert parameters to complex tensors
        omega_c = torch.tensor(self.omega, dtype=torch.complex128, device=self.device)
        alpha_c = torch.tensor(self.alpha, dtype=torch.complex128, device=self.device)
        beta_c = torch.tensor(self.beta, dtype=torch.complex128, device=self.device)
        gamma_c = torch.tensor(self.gamma, dtype=torch.complex128, device=self.device)
        lambda_c = torch.tensor(self.lambda_, dtype=torch.complex128, device=self.device)
        r_daily_c = torch.tensor(self.r_daily, dtype=torch.complex128, device=self.device)
        
        # Initialize coefficient storage
        # Shape: [N+1, N_quad, 2, 3]
        # Dimension 2 (size 2): const=1.0 and const=0.0
        # Dimension 3 (size 3): [coeff_K, coeff_S, const_term]
        coefficients = torch.zeros(
            (N+1, self.N_quad, 2, 3),
            dtype=torch.complex128,
            device=self.device
        )
        
        # Compute auxiliary parameters
        lambda_r = torch.tensor(-0.5, dtype=torch.complex128, device=self.device)
        gamma_r = gamma_c + lambda_c + 0.5
        denom_sigma = 1.0 - beta_c - alpha_c * gamma_r**2
        sigma2 = (omega_c + alpha_c) / denom_sigma
        
        # Vectorize over quadrature nodes
        u_vec = torch.tensor(u_nodes, dtype=torch.complex128, device=self.device)
        cphi0_vec = 1j * u_vec  # [N_quad]
        
        # Compute coefficients for each time step
        for n in range(N+1):
            Time_inDays = N - n
            
            # Process both const=1.0 and const=0.0
            for const_idx, const_val in enumerate([1.0, 0.0]):
                const_c = torch.tensor(const_val, dtype=torch.complex128, device=self.device)
                cphi_vec = cphi0_vec + const_c  # [N_quad]
                
                # Initialize recursion
                a_vec = cphi_vec * r_daily_c  # [N_quad]
                b_vec = lambda_r * cphi_vec + 0.5 * cphi_vec**2  # [N_quad]
                
                # Recursive calculation (vectorized over N_quad)
                for i in range(1, Time_inDays):
                    denom_vec = 1.0 - 2.0 * alpha_c * b_vec
                    a_vec = (
                        a_vec + cphi_vec * r_daily_c + b_vec * omega_c -
                        0.5 * torch.log(denom_vec)
                    )
                    b_vec = (
                        cphi_vec * (lambda_r + gamma_r) - 0.5 * gamma_r**2 +
                        beta_c * b_vec +
                        0.5 * (cphi_vec - gamma_r)**2 / denom_vec
                    )
                
                # Store results
                coefficients[n, :, const_idx, 0] = -cphi0_vec
                coefficients[n, :, const_idx, 1] = cphi_vec
                coefficients[n, :, const_idx, 2] = a_vec + b_vec * sigma2
        
        logger.info(f"Successfully precomputed coefficients for N={N}")
        
        return {
            "coefficients": coefficients,
            "u_nodes": u_nodes_t,
            "w_nodes": w_nodes_t,
            "N": N,
            "r_daily": self.r_daily,
            "device": str(self.device)
        }


class PrecomputationManager:
    """
    Manages precomputation for multiple instrument maturities.
    
    This class loads precomputed coefficients from disk (precomputed_cache/)
    instead of computing them on the fly.
    """
    
    def __init__(
        self,
        garch_params: Dict[str, float],
        r_annual: float,
        maturities: list,
        N_quad: int = 128,
        u_max: float = 100.0,
        device: str = "cpu",
        cache_dir: str = "precomputed_cache"
    ):
        """
        Initialize the precomputation manager.
        
        Args:
            garch_params: GARCH model parameters
            r_annual: Annual risk-free rate
            maturities: List of maturities in days to load
            N_quad: Number of quadrature points (must match precomputed data)
            u_max: Maximum integration range (must match precomputed data)
            device: Computation device
            cache_dir: Directory containing precomputed data files
        """
        self.garch_params = garch_params
        self.r_annual = r_annual
        self.r_daily = r_annual / 252.0
        self.maturities = sorted(set(maturities))  # Remove duplicates and sort
        self.N_quad = N_quad
        self.u_max = u_max
        self.device = device
        self.cache_dir = Path(cache_dir)
        
        self.precomputed_data = {}
        
        # Validate cache directory exists
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}\n"
                f"Please run precompute_all_maturities.py first to generate the cache."
            )
        
        logger.info(
            f"Initialized PrecomputationManager for maturities: {self.maturities}"
        )
        logger.info(f"Loading from cache directory: {self.cache_dir}")
    
    def _load_from_disk(self, maturity: int) -> Dict[str, Any]:
        """
        Load precomputed data for a specific maturity from disk.
        
        Args:
            maturity: Maturity in days
            
        Returns:
            Dictionary containing precomputed coefficients and metadata
        """
        filename = self.cache_dir / f"precomputed_N{maturity:04d}.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(
                f"Precomputed data not found: {filename}\n"
                f"Available range: N0001.pkl to N0504.pkl\n"
                f"Please run: python precompute_all_maturities.py --min-maturity {maturity} --max-maturity {maturity}"
            )
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded precomputed data for N={maturity} from {filename}")
        
        # Move tensors to the correct device
        if 'coefficients' in data:
            data['coefficients'] = data['coefficients'].to(self.device)
        if 'u_nodes' in data:
            data['u_nodes'] = data['u_nodes'].to(self.device)
        if 'w_nodes' in data:
            data['w_nodes'] = data['w_nodes'].to(self.device)
        
        # Validate parameters match
        if data.get('N') != maturity:
            logger.warning(
                f"Maturity mismatch: requested {maturity}, file contains {data.get('N')}"
            )
        
        return data
    
    def precompute_all(self) -> Dict[int, Dict[str, Any]]:
        """
        Load precomputed coefficients for all required maturities from disk.
        
        Returns:
            Dictionary mapping maturity (in days) to precomputed data
        """
        logger.info("Loading precomputed data for all maturities...")
        
        for maturity in self.maturities:
            try:
                logger.info(f"Loading maturity: {maturity} days")
                self.precomputed_data[maturity] = self._load_from_disk(maturity)
            except FileNotFoundError as e:
                logger.error(str(e))
                raise
            except Exception as e:
                logger.error(f"Failed to load maturity {maturity}: {e}")
                raise
        
        logger.info(f"Successfully loaded precomputed data for {len(self.precomputed_data)} maturities")
        return self.precomputed_data
    
    def get_precomputed_data(self, maturity: int) -> Optional[Dict[str, Any]]:
        """
        Get precomputed data for a specific maturity.
        
        If not already loaded, attempts to load from disk.
        
        Args:
            maturity: Maturity in days
            
        Returns:
            Precomputed data dictionary or None if not found
        """
        if maturity not in self.precomputed_data:
            logger.info(f"Maturity {maturity} not in cache, loading from disk...")
            try:
                self.precomputed_data[maturity] = self._load_from_disk(maturity)
            except FileNotFoundError:
                logger.warning(f"No precomputed data found for maturity {maturity}")
                return None
        
        return self.precomputed_data[maturity]
    
    def precompute_for_maturity(self, maturity: int) -> None:
        """
        Load precomputed data for a specific maturity (convenience method).
        
        Args:
            maturity: Maturity in days
        """
        if maturity not in self.maturities:
            self.maturities.append(maturity)
            self.maturities = sorted(self.maturities)
        
        self.precomputed_data[maturity] = self._load_from_disk(maturity)


def create_precomputation_manager_from_config(
    config: Dict[str, Any]
) -> PrecomputationManager:
    """
    Factory function to create a PrecomputationManager from configuration.
    
    Args:
        config: Configuration dictionary containing:
            - garch: GARCH parameters
            - simulation: Simulation parameters (for r)
            - instruments: Instrument configuration (for maturities)
            - precomputation: Precomputation parameters
            
    Returns:
        Configured PrecomputationManager instance
    """
    # Get cache directory from config or use default
    cache_dir = config.get("precomputation", {}).get("cache_dir", "precomputed_cache")
    
    manager = PrecomputationManager(
        garch_params=config["garch"],
        r_annual=config["simulation"]["r"],
        maturities=config["instruments"]["maturities"],
        N_quad=config["precomputation"]["N_quad"],
        u_max=config["precomputation"]["u_max"],
        device=config["precomputation"]["device"],
        cache_dir=cache_dir
    )
    
    return manager
