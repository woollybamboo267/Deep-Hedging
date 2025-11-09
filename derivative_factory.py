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
            # Convert hedged option maturity from years to days
            maturity_days = int(hedged_cfg['T'] * 252)
            
            if maturity_days not in precomputed_data:
                raise ValueError(
                    f"No precomputed data for hedged vanilla option maturity {maturity_days} days. "
                    f"Available maturities: {list(precomputed_data.keys())}"
                )
            
            # Create a minimal PrecomputationManager wrapper
            class PrecompManagerWrapper:
                def __init__(self, precomp_dict, r_daily, mat_days):
                    self.r_daily = r_daily
                    self._data = {mat_days: precomp_dict}
                
                def get_precomputed_data(self, N):
                    return self._data.get(N)
            
            precomp_manager = PrecompManagerWrapper(
                precomputed_data[maturity_days],
                config['simulation']['r'] / 252.0,
                maturity_days
            )
            
            logger.info(f"Created vanilla hedged option with maturity {maturity_days} days")
            
            vanilla_option = VanillaOption(
                precomputation_manager=precomp_manager,
                garch_params=config['garch'],
                option_type=hedged_cfg['option_type']
            )
            
            # FIX: Store maturity and strike on the option object
            vanilla_option.N = maturity_days
            vanilla_option.K = hedged_cfg['K']
            
            return vanilla_option
        
        elif deriv_type == 'barrier':
            # Use BarrierOption with neural network (no precomputation needed)
            logger.info(
                f"Created barrier hedged option: {hedged_cfg['barrier_type']} "
                f"{hedged_cfg['option_type']} with barrier={hedged_cfg['barrier_level']}"
            )
            
            barrier_option = BarrierOption(
                model_path=hedged_cfg['model_path'],
                barrier_level=hedged_cfg['barrier_level'],
                barrier_type=hedged_cfg.get('barrier_type', 'up-and-in'),
                option_type=hedged_cfg['option_type'],
                r_annual=config['simulation']['r'],
                device=config['training']['device']
            )
            
            # FIX: Store maturity and strike on barrier option too
            barrier_option.N = int(hedged_cfg['T'] * 252)
            barrier_option.K = hedged_cfg['K']
            
            return barrier_option
        
        else:
            raise ValueError(f"Unknown derivative type: {deriv_type}")
    
    @staticmethod
    def create_hedging_derivatives(
        config: Dict[str, Any],
        precomputed_data: Dict[int, Dict[str, Any]]
    ) -> List:
        """
        Create hedging instruments (always vanilla).
        
        Hedging instruments structure:
        - Instrument 0: Stock (represented as None)
        - Instruments 1+: Vanilla options with strikes/maturities from config
        
        Args:
            config: Full config dict
            precomputed_data: Dict mapping maturity to precomputed coefficients
        
        Returns:
            List of derivative objects [None (stock), VanillaOption1, VanillaOption2, ...]
        """
        instruments_cfg = config['instruments']
        n_instruments = instruments_cfg['n_hedging_instruments']
        
        # First instrument is always stock (None)
        hedging_derivs = [None]
        
        # If only hedging with stock, return early
        if n_instruments == 1:
            logger.info("Hedging with stock only (no vanilla options)")
            return hedging_derivs
        
        # Get hedging option parameters
        strikes = instruments_cfg['strikes']  # Length: n_instruments - 1
        option_types = instruments_cfg['types']  # Length: n_instruments - 1
        maturities = instruments_cfg['maturities']  # Length: n_instruments - 1
        
        # Validate lengths
        n_options = n_instruments - 1
        if len(strikes) != n_options:
            raise ValueError(f"Expected {n_options} strikes, got {len(strikes)}")
        if len(option_types) != n_options:
            raise ValueError(f"Expected {n_options} option types, got {len(option_types)}")
        if len(maturities) != n_options:
            raise ValueError(f"Expected {n_options} maturities, got {len(maturities)}")
        
        # Create vanilla hedging options
        for i in range(n_options):
            maturity_days = maturities[i]
            strike = strikes[i]
            opt_type = option_types[i]
            
            if maturity_days not in precomputed_data:
                raise ValueError(
                    f"No precomputed data for hedging maturity {maturity_days} days. "
                    f"Available maturities: {list(precomputed_data.keys())}"
                )
            
            # Create PrecomputationManager wrapper for this maturity
            class PrecompManagerWrapper:
                def __init__(self, precomp_dict, r_daily, mat_days):
                    self.r_daily = r_daily
                    self._data = {mat_days: precomp_dict}
                
                def get_precomputed_data(self, N):
                    return self._data.get(N)
            
            precomp_manager = PrecompManagerWrapper(
                precomputed_data[maturity_days],
                config['simulation']['r'] / 252.0,
                maturity_days
            )
            
            hedging_option = VanillaOption(
                precomputation_manager=precomp_manager,
                garch_params=config['garch'],
                option_type=opt_type
            )
            
            # FIX: Store maturity and strike on the hedging option object
            hedging_option.N = maturity_days
            hedging_option.K = strike
            
            hedging_derivs.append(hedging_option)
            
            logger.info(
                f"Created hedging option {i+1}: {opt_type} with K={strike}, "
                f"T={maturity_days} days"
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
    logger.info(f"Available precomputed maturities: {list(precomputed_data.keys())}")
    
    # Create hedged derivative
    hedged_derivative = DerivativeFactory.create_hedged_derivative(config, precomputed_data)
    logger.info(f"Created hedged derivative: {type(hedged_derivative).__name__}")
    
    # Create hedging derivatives
    hedging_derivatives = DerivativeFactory.create_hedging_derivatives(config, precomputed_data)
    logger.info(f"Created {len(hedging_derivatives)} hedging instruments")
    
    for i, deriv in enumerate(hedging_derivatives):
        if deriv is None:
            logger.info(f"  Instrument {i}: Stock")
        else:
            logger.info(f"  Instrument {i}: {type(deriv).__name__}")
    
    return hedged_derivative, hedging_derivatives
