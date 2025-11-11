import torch
from typing import Optional
from src.option_greek.base import DerivativeBase
from src.option_greek.vanilla import VanillaOption
from src.option_greek.barrier import BarrierOption


class BarrierOptionWithVanillaFallback(DerivativeBase):
    """
    Wrapper for up-and-in barrier options with vanilla fallback after breach.
    
    Before breach: Uses barrier neural network pricing (accounts for breach probability)
    After breach: Switches to vanilla option analytical pricing
    """
    
    def __init__(
        self,
        barrier_option: BarrierOption,
        vanilla_option: VanillaOption
    ):
        """
        Initialize barrier wrapper.
        
        Args:
            barrier_option: BarrierOption instance for pre-breach pricing
            vanilla_option: VanillaOption instance for post-breach pricing
        """
        self.barrier_option = barrier_option
        self.vanilla_option = vanilla_option
        self.barrier_level = barrier_option.barrier_level
        self.option_type = barrier_option.option_type
        self.K = None
        self.barrier_breached = False
    
    def reset_barrier_status(self):
        """Reset barrier breach status to False."""
        self.barrier_breached = False
    
    def check_and_update_barrier(self, S: torch.Tensor) -> bool:
        """
        Check if current price breaches barrier and update status.
        
        Args:
            S: Current spot price(s)
            
        Returns:
            True if barrier was just breached, False otherwise
        """
        if isinstance(S, torch.Tensor):
            S_max = S.max().item()
        else:
            S_max = float(S)
        
        if not self.barrier_breached and S_max >= self.barrier_level:
            self.barrier_breached = True
            return True
        
        return False
    
    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """
        Price up-and-in barrier option.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance
            
        Returns:
            Option price
        """
        self.check_and_update_barrier(S)
        self.K = K
        
        if not self.barrier_breached:
            return self.barrier_option.price(S, K, step_idx, N, h0)
        else:
            return self.vanilla_option.price(S, K, step_idx, N)
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute delta."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if not self.barrier_breached:
            return self.barrier_option.delta(S, K, step_idx, N, h0)
        else:
            return self.vanilla_option.delta(S, K, step_idx, N)
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute gamma."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if not self.barrier_breached:
            return self.barrier_option.gamma(S, K, step_idx, N, h0)
        else:
            return self.vanilla_option.gamma(S, K, step_idx, N)
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute vega."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if not self.barrier_breached:
            return self.barrier_option.vega(S, K, step_idx, N, h0)
        else:
            return self.vanilla_option.vega(S, K, step_idx, N)
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute theta."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if not self.barrier_breached:
            return self.barrier_option.theta(S, K, step_idx, N, h0)
        else:
            return self.vanilla_option.theta(S, K, step_idx, N)
