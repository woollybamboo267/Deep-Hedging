"""
Factory for creating derivative objects from config.
Supports vanilla, barrier, and American options.
"""

from typing import Dict, Any, List
import torch
import logging
from src.option_greek.vanilla import VanillaOption
from src.option_greek.barrier import BarrierOption
from src.option_greek.american import AmericanOption
from src.option_greek.precompute import PrecomputationManager
from src.option_greek.barrier_wrapper import BarrierOptionWithVanillaFallback

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
                             Only required for vanilla and barrier options
        
        Returns:
            VanillaOption, BarrierOptionWithVanillaFallback, or AmericanOption instance
        """
        hedged_cfg = config['hedged_option']
        deriv_type = hedged_cfg['type'].lower()
        
        # === VANILLA OPTION ===
        if deriv_type == 'vanilla':
            maturity_days = int(hedged_cfg['T'] * 252)
            
            if maturity_days not in precomputed_data:
                raise ValueError(
                    f"No precomputed data for hedged vanilla option maturity {maturity_days} days. "
                    f"Available maturities: {list(precomputed_data.keys())}"
                )
            
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
            
            vanilla_option.N = maturity_days
            vanilla_option.K = hedged_cfg['K']
            
            return vanilla_option
        
        # === BARRIER OPTION ===
        elif deriv_type == 'barrier':
            logger.info(
                f"Creating barrier hedged option ({hedged_cfg['option_type']}) "
                f"with barrier={hedged_cfg['barrier_level']}"
            )
            
            # Create barrier option (model-driven)
            barrier_option = BarrierOption(
                model_path=hedged_cfg['model_path'],
                barrier_level=hedged_cfg['barrier_level'],
                option_type=hedged_cfg['option_type'],
                r_annual=config['simulation']['r'],
                device=config['training']['device']
            )
            
            barrier_option.N = int(hedged_cfg['T'] * 252)
            barrier_option.K = hedged_cfg['K']
            
            maturity_days = int(hedged_cfg['T'] * 252)
            
            if maturity_days not in precomputed_data:
                raise ValueError(
                    f"No precomputed vanilla data for barrier fallback at maturity {maturity_days} days. "
                    f"Barrier options require vanilla coefficients for post-breach pricing."
                )
            
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
            
            # Create vanilla fallback
            vanilla_fallback = VanillaOption(
                precomputation_manager=precomp_manager,
                garch_params=config['garch'],
                option_type=hedged_cfg['option_type']
            )
            vanilla_fallback.N = maturity_days
            vanilla_fallback.K = hedged_cfg['K']
            
            # Wrap barrier + vanilla fallback
            wrapped_barrier = BarrierOptionWithVanillaFallback(
                barrier_option=barrier_option,
                vanilla_option=vanilla_fallback
            )
            
            logger.info("Wrapped barrier option with vanilla fallback for breach handling")
            
            return wrapped_barrier
        
        # === AMERICAN OPTION ===
        elif deriv_type == 'american':
            logger.info(
                f"Creating American hedged option ({hedged_cfg['option_type']}) "
                f"with strike K={hedged_cfg['K']}"
            )
            
            # Create American option (model-driven, no precomputation needed)
            american_option = AmericanOption(
                model_path=hedged_cfg['model_path'],
                option_type=hedged_cfg['option_type'],
                r_annual=config['simulation']['r'],
                device=config['training']['device']
            )
            
            american_option.N = int(hedged_cfg['T'] * 252)
            american_option.K = hedged_cfg['K']
            
            logger.info(f"Created American option with maturity {american_option.N} days")
            
            return american_option
        
        else:
            raise ValueError(f"Unknown derivative type: {deriv_type}")
    
    # === HEDGING DERIVATIVES ===
    @staticmethod
    def create_hedging_derivatives(
        config: Dict[str, Any],
        precomputed_data: Dict[int, Dict[str, Any]]
    ) -> List:
        """
        Create hedging instruments (always vanilla options).
        
        Args:
            config: Full config dict
            precomputed_data: Dict mapping maturity (in days) to precomputed coefficients
                             Only required if using vanilla options as hedging instruments
        
        Returns:
            List of derivative objects [None (stock), VanillaOption1, VanillaOption2, ...]
        """
        instruments_cfg = config['instruments']
        n_instruments = instruments_cfg['n_hedging_instruments']
        
        hedging_derivs = [None]
        
        if n_instruments == 1:
            logger.info("Hedging with stock only (no options)")
            return hedging_derivs
        
        strikes = instruments_cfg['strikes']
        option_types = instruments_cfg['types']
        maturities = instruments_cfg['maturities']
        
        n_options = n_instruments - 1
        if len(strikes) != n_options:
            raise ValueError(f"Expected {n_options} strikes, got {len(strikes)}")
        if len(option_types) != n_options:
            raise ValueError(f"Expected {n_options} option types, got {len(option_types)}")
        if len(maturities) != n_options:
            raise ValueError(f"Expected {n_options} maturities, got {len(maturities)}")
        
        for i in range(n_options):
            maturity_days = maturities[i]
            strike = strikes[i]
            opt_type = option_types[i]
            
            if maturity_days not in precomputed_data:
                raise ValueError(
                    f"No precomputed data for hedging maturity {maturity_days} days. "
                    f"Available maturities: {list(precomputed_data.keys())}"
                )
            
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
    
    Args:
        config: Full config dict
        precomputed_data: Dict mapping maturity (in days) to precomputed coefficients
                         Only required for vanilla options (hedged or hedging)
    
    Returns:
        (hedged_derivative, hedging_derivatives_list)
    """
    logger.info("Setting up derivatives from config...")
    logger.info(f"Available precomputed maturities: {list(precomputed_data.keys())}")
    
    hedged_derivative = DerivativeFactory.create_hedged_derivative(config, precomputed_data)
    logger.info(f"Created hedged derivative: {type(hedged_derivative).__name__}")
    
    hedging_derivatives = DerivativeFactory.create_hedging_derivatives(config, precomputed_data)
    logger.info(f"Created {len(hedging_derivatives)} hedging instruments")
    
    for i, deriv in enumerate(hedging_derivatives):
        if deriv is None:
            logger.info(f"  Instrument {i}: Stock")
        else:
            logger.info(f"  Instrument {i}: {type(deriv).__name__}")
    
    return hedged_derivative, hedging_derivatives
