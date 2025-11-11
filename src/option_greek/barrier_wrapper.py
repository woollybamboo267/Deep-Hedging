import torch
from typing import Optional
from src.option_greek.base import DerivativeBase
from src.option_greek.vanilla import VanillaOption
from src.option_greek.barrier import BarrierOption


class BarrierOptionWithVanillaFallback(DerivativeBase):
    """
    Wrapper for up-and-in barrier options with vanilla fallback after breach.
    
    CRITICAL FIX: This version tracks barrier breach status PER PATH.
    The original version used a single boolean flag for all paths, causing
    massive hedging errors when some paths breached and others didn't.
    
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
        
        # Path-specific breach tracking (CRITICAL FIX)
        self.barrier_breached_mask = None  # Will be [M] boolean tensor
        self.num_paths = None
    
    def reset_barrier_status(self, num_paths: int = None):
        """
        Reset barrier breach status for all paths.
        
        Args:
            num_paths: Number of simulation paths (required for proper initialization)
        """
        if num_paths is not None:
            self.num_paths = num_paths
            device = self.barrier_option.device if hasattr(self.barrier_option, 'device') else 'cpu'
            self.barrier_breached_mask = torch.zeros(
                num_paths, dtype=torch.bool, device=device
            )
        else:
            self.barrier_breached_mask = None
            self.num_paths = None
    
    def check_and_update_barrier(self, S: torch.Tensor) -> torch.Tensor:
        """
        Check if barrier is breached for each path independently.
        
        CRITICAL: This now returns a per-path boolean mask instead of a single boolean.
        
        Args:
            S: Current spot price(s) - shape [M] or [M, ...]
            
        Returns:
            Boolean tensor [M] indicating which paths have breached the barrier
        """
        # Handle scalar input
        if S.dim() == 0:
            S = S.unsqueeze(0)
        
        # Initialize mask if this is the first call
        if self.barrier_breached_mask is None:
            self.num_paths = S.shape[0]
            self.barrier_breached_mask = torch.zeros(
                self.num_paths, dtype=torch.bool, device=S.device
            )
        
        # Check barrier breach per path
        # If S has more than 1 dimension, take max along non-batch dimensions
        if S.dim() > 1:
            S_check = S.reshape(S.shape[0], -1).max(dim=1)[0]
        else:
            S_check = S
        
        # Update breach status: once breached, stays breached
        newly_breached = S_check >= self.barrier_level
        self.barrier_breached_mask = self.barrier_breached_mask | newly_breached
        
        return self.barrier_breached_mask
    
    def _compute_mixed_greek(
        self, 
        S: torch.Tensor, 
        K: float, 
        step_idx: int, 
        N: int, 
        h0: float,
        greek_name: str
    ) -> torch.Tensor:
        """
        Compute Greek with path-specific barrier handling.
        
        This is the core fix: compute both barrier and vanilla Greeks,
        then use torch.where() to select the correct value per path.
        
        Args:
            S: Spot prices [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance
            greek_name: 'price', 'delta', 'gamma', 'vega', or 'theta'
            
        Returns:
            Greek values with correct formula per path
        """
        # Update barrier breach status
        breach_mask = self.check_and_update_barrier(S)
        self.K = K
        
        # Get the appropriate method from each model
        barrier_method = getattr(self.barrier_option, greek_name)
        vanilla_method = getattr(self.vanilla_option, greek_name)
        
        # Compute both barrier and vanilla values
        barrier_values = barrier_method(S, K, step_idx, N, h0)
        vanilla_values = vanilla_method(S, K, step_idx, N)
        
        # Ensure both have the same shape
        if barrier_values.shape != vanilla_values.shape:
            raise ValueError(
                f"Shape mismatch: barrier={barrier_values.shape}, "
                f"vanilla={vanilla_values.shape}"
            )
        
        # Broadcast breach mask to match value dimensions if needed
        mask = breach_mask
        while mask.dim() < barrier_values.dim():
            mask = mask.unsqueeze(-1)
        
        # Select: vanilla where breached, barrier where not breached
        result = torch.where(mask, vanilla_values, barrier_values)
        
        return result
    
    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """
        Price up-and-in barrier option with path-specific barrier handling.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance
            
        Returns:
            Option price per path
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'price')
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """
        Compute delta with path-specific barrier handling.
        
        CRITICAL: Paths that breached use vanilla delta, paths that haven't use barrier delta.
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'delta')
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute gamma with path-specific barrier handling."""
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'gamma')
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute vega with path-specific barrier handling."""
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'vega')
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float = None, **kwargs) -> torch.Tensor:
        """Compute theta with path-specific barrier handling."""
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'theta')
