import torch
from typing import Optional
from .base import DerivativeBase
from .vanilla import VanillaOption
from .barrier import BarrierOption


class BarrierOptionWithVanillaFallback(DerivativeBase):
    """
    Wrapper that handles barrier breach logic by switching to vanilla pricing.
    
    For up-and-in options: price=0 before breach, vanilla after breach
    For up-and-out options: vanilla before breach, price=0 after breach
    """
    
    def __init__(
        self,
        barrier_option: BarrierOption,
        vanilla_option: VanillaOption,
        barrier_type: str
    ):
        """
        Initialize barrier wrapper with both pricers.
        
        Args:
            barrier_option: BarrierOption instance (not used after breach detection)
            vanilla_option: VanillaOption instance for post-breach pricing
            barrier_type: 'up-and-in' or 'up-and-out'
        """
        self.barrier_option = barrier_option
        self.vanilla_option = vanilla_option
        self.barrier_type = barrier_type
        self.barrier_level = barrier_option.barrier_level
        self.option_type = barrier_option.option_type
        self.K = None
        
        self.barrier_breached = False
    
    def reset_barrier_status(self):
        """Reset barrier breach status to False (call at start of new path)."""
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
        Price with barrier breach logic.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (for compatibility)
            **kwargs: Additional arguments
        """
        self.check_and_update_barrier(S)
        self.K = K
        
        if self.barrier_type == "up-and-in":
            if not self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.price(S, K, step_idx, N)
        
        elif self.barrier_type == "up-and-out":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.price(S, K, step_idx, N)
        
        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute delta with barrier breach handling."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if self.barrier_type == "up-and-in":
            if not self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.delta(S, K, step_idx, N)
        
        elif self.barrier_type == "up-and-out":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.delta(S, K, step_idx, N)
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute gamma with barrier breach handling."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if self.barrier_type == "up-and-in":
            if not self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.gamma(S, K, step_idx, N)
        
        elif self.barrier_type == "up-and-out":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.gamma(S, K, step_idx, N)
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute vega with barrier breach handling."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if self.barrier_type == "up-and-in":
            if not self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.vega(S, K, step_idx, N)
        
        elif self.barrier_type == "up-and-out":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.vega(S, K, step_idx, N)
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute theta with barrier breach handling."""
        self.check_and_update_barrier(S)
        self.K = K
        
        if self.barrier_type == "up-and-in":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.theta(S, K, step_idx, N)
        
        elif self.barrier_type == "up-and-out":
            if self.barrier_breached:
                return torch.zeros_like(torch.as_tensor(S, dtype=torch.float32))
            else:
                return self.vanilla_option.theta(S, K, step_idx, N)
