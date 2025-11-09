"""
Factory for creating derivative objects from config.
Supports vanilla and barrier options.
"""

from typing import Dict, Any, List, Optional
import torch
import logging
from src.option_greek.vanilla import VanillaOption
from src.option_greek.barrier import BarrierOption
from src.option_greek.precompute import PrecomputationManager

logger = logging.getLogger(__name__)


class DerivativeFactory:
    """Factory for creating derivative objects based on config."""
    
    @staticmethod
    def create_hedged_derivative(
        config: Dict[str, Any], 
        precomputed_data: Dict[int, Dict[str, Any]]
    ) -> Any:
        """
        Create the derivative to be hedged.
        
        Args:
            config: Full config dict
            precomputed_data: Dict mapping maturity (in days) to precomputed coefficients
        
        Returns:
            VanillaOption or BarrierOption instance
        """
        hedged_cfg = config['hedged_option']
        deriv_type = hedged_cfg['type'].lower()
        
        if deriv_type == 'vanilla':
            # Use VanillaOption with precomputed data
            maturity_days = config['simulation']['N']  # Should be 252 for 1-year
            
            if maturity_days not in precomputed_data:
                raise ValueError(f"No precomputed data for maturity {maturity_days}")
            
            # Create a minimal PrecomputationManager wrapper
            class PrecompManagerWrapper:
                def __init__(self, precomp_dict, r_daily):
                    self.r_daily = r_daily
                    self._data = {maturity_days: precomp_dict}
                
                def get_precomputed_data(self, N):
                    return self._data.get(N)
            
            precomp_manager = PrecompManagerWrapper(
                precomputed_data[maturity_days],
                config['simulation']['r'] / 252.0
            )
            
            return VanillaOption(
                precomputation_manager=precomp_manager,
                garch_params=config['garch'],
                option_type=hedged_cfg['option_type']
            )
        
        elif deriv_type == 'barrier':
            # Use BarrierOption with neural network
            return BarrierOption(
                model_path=hedged_cfg['model_path'],
                barrier_level=hedged_cfg['barrier_level'],
                barrier_type=hedged_cfg.get('barrier_type', 'up-and-in'),
                option_type=hedged_cfg['option_type'],
                r_annual=config['simulation']['r'],
                device=config['training']['device']
            )
        
        else:
            raise ValueError(f"Unknown derivative type: {deriv_type}")
    
    @staticmethod
    def create_hedging_derivatives(
        config: Dict[str, Any],
        precomputed_data: Dict[int, Dict[str, Any]]
    ) -> List:
        """
        Create hedging instruments.
        
        Args:
            config: Full config dict
            precomputed_data: Dict mapping maturity to precomputed coefficients
        
        Returns:
            List of derivative objects [None (stock), Option1, Option2, ...]
        """
        instruments_cfg = config['instruments']
        n_instruments = instruments_cfg['n_hedging_instruments']
        
        # First instrument is always stock (None)
        hedging_derivs = [None]
        
        if n_instruments == 1:
            return hedging_derivs
        
        # Get strikes and types
        strikes = instruments_cfg.get('strikes', [])
        option_types = instruments_cfg.get('types', [])
        maturities = instruments_cfg.get('maturities', [252, 504])
        
        # Skip first maturity (252) as it's for the hedged option
        hedge_maturities = maturities[1:]
        
        # Create hedging options
        for i, maturity_days in enumerate(hedge_maturities):
            strike = strikes[i] if i < len(strikes) else config['simulation']['S0']
            opt_type = option_types[i] if i < len(option_types) else 'call'
            
            if maturity_days not in precomputed_data:
                raise ValueError(f"No precomputed data for hedging maturity {maturity_days}")
            
            # Create PrecomputationManager wrapper for this maturity
            class PrecompManagerWrapper:
                def __init__(self, precomp_dict, r_daily, mat_days):
                    self.r_daily = r_daily
                    self._data = {mat_days: precomp_dict}
                    self.N = mat_days
                
                def get_precomputed_data(self, N):
                    return self._data.get(N)
            
            precomp_manager = PrecompManagerWrapper(
                precomputed_data[maturity_days],
                config['simulation']['r'] / 252.0,
                maturity_days
            )
            
            hedging_derivs.append(
                VanillaOption(
                    precomputation_manager=precomp_manager,
                    garch_params=config['garch'],
                    option_type=opt_type
                )
            )
        
        return hedging_derivs


def setup_derivatives_from_precomputed(
    config: Dict[str, Any],
    precomputed_data: Dict[int, Dict[str, Any]]
) -> tuple:
    """
    Setup all derivatives from config and existing precomputed data.
    
    This function is called AFTER precomputation is done in train.py.
    
    Args:
        config: Full config dict
        precomputed_data: Dict mapping maturity (in days) to precomputed coefficients
    
    Returns:
        (hedged_derivative, hedging_derivatives_list)
    """
    logger.info("Setting up derivatives from config...")
    
    # Create hedged derivative
    hedged_derivative = DerivativeFactory.create_hedged_derivative(config, precomputed_data)
    logger.info(f"Created hedged derivative: {type(hedged_derivative).__name__}")
    
    # Create hedging derivatives
    hedging_derivatives = DerivativeFactory.create_hedging_derivatives(config, precomputed_data)
    logger.info(f"Created {len(hedging_derivatives)} hedging instruments")
    
    for i, deriv in enumerate(hedging_derivatives):
        if deriv is None:
            logger.info(f"  [{i}] Stock")
        else:
            logger.info(f"  [{i}] {type(deriv).__name__}")
    
    return hedged_derivative, hedging_derivatives
