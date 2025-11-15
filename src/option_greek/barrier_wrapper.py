import torch
from typing import Optional
from src.option_greek.base import DerivativeBase
from src.option_greek.vanilla import VanillaOption
from src.option_greek.barrier import BarrierOption


class BarrierOptionWithVanillaFallback(DerivativeBase):
    """
    Wrapper for up-and-in barrier options with vanilla fallback after breach.
    
    CRITICAL FIX: This version tracks barrier breach status PER PATH and properly
    handles autograd for barrier parameter sensitivities.
    
    Before breach: Uses barrier neural network pricing (accounts for breach probability)
    After breach: Switches to vanilla option analytical pricing
    
    KEY FIXES:
    1. Per-path barrier breach tracking (not single boolean)
    2. Proper autograd handling for barrier_level parameter
    3. Correct batched parameter handling without dimension mismatches
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
        
        # Ensure barrier_level is a scalar tensor with requires_grad for autograd
        if not isinstance(barrier_option.barrier_level, torch.Tensor):
            self.barrier_level = torch.tensor(
                barrier_option.barrier_level, 
                dtype=torch.float32,
                device=barrier_option.device,
                requires_grad=True
            )
            # Update the barrier_option's barrier_level to point to this tensor
            self.barrier_option.barrier_level = self.barrier_level
        else:
            self.barrier_level = barrier_option.barrier_level
            
        self.option_type = barrier_option.option_type
        self.K = None
        
        # Path-specific breach tracking (CRITICAL FIX)
        self.barrier_breached_mask = None  # Will be [M] boolean tensor
        self.num_paths = None
    
    @property
    def device(self):
        """Get device from barrier option."""
        return self.barrier_option.device if hasattr(self.barrier_option, 'device') else 'cpu'
    
    def reset_barrier_status(self, num_paths: int = None):
        """
        Reset barrier breach status for all paths.
        
        Args:
            num_paths: Number of simulation paths (required for proper initialization)
        """
        if num_paths is not None:
            self.num_paths = num_paths
            self.barrier_breached_mask = torch.zeros(
                num_paths, dtype=torch.bool, device=self.device
            )
        else:
            self.barrier_breached_mask = None
            self.num_paths = None
    
    def check_and_update_barrier(self, S: torch.Tensor) -> torch.Tensor:
        """
        Check if barrier is breached for each path independently.
        
        CRITICAL: Returns per-path boolean mask, not single boolean.
        Uses detached barrier_level for comparison to avoid breaking autograd graph.
        
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
        
        # CRITICAL: Use detached barrier_level for comparison
        # Comparison operations don't need gradients and can break autograd
        barrier_threshold = self.barrier_level.detach() if isinstance(self.barrier_level, torch.Tensor) else self.barrier_level
        
        # Update breach status: once breached, stays breached
        newly_breached = S_check >= barrier_threshold
        self.barrier_breached_mask = self.barrier_breached_mask | newly_breached
        
        return self.barrier_breached_mask
    
    def _compute_mixed_greek(
        self, 
        S: torch.Tensor, 
        K: float, 
        step_idx: int, 
        N: int, 
        h0: Optional[float],
        greek_name: str
    ) -> torch.Tensor:
        """
        Compute Greek with path-specific barrier handling and proper autograd support.
        
        This is the core fix: compute both barrier and vanilla Greeks,
        then use torch.where() to select the correct value per path.
        
        AUTOGRAD FIX: Pass h0 as scalar to barrier methods, letting PyTorch
        handle broadcasting internally. This avoids dimension mismatches.
        
        Args:
            S: Spot prices [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (scalar or None)
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
        
        # Compute barrier values
        # CRITICAL FIX: Pass h0 as scalar, not batched tensor
        # PyTorch will handle broadcasting internally during feature creation
        if h0 is not None:
            # Ensure h0 is a scalar tensor if it needs gradients
            if not isinstance(h0, torch.Tensor):
                h0_tensor = torch.tensor(h0, dtype=torch.float32, device=self.device)
            else:
                h0_tensor = h0
            barrier_values = barrier_method(S, K, step_idx, N, h0_tensor)
        else:
            barrier_values = barrier_method(S, K, step_idx, N)
        
        # Compute vanilla values (doesn't need h0)
        vanilla_values = vanilla_method(S, K, step_idx, N)
        
        # Ensure both have the same shape
        if barrier_values.shape != vanilla_values.shape:
            # Try to broadcast if dimensions don't match
            try:
                barrier_values = barrier_values.expand_as(vanilla_values)
            except RuntimeError:
                try:
                    vanilla_values = vanilla_values.expand_as(barrier_values)
                except RuntimeError:
                    raise ValueError(
                        f"Shape mismatch that cannot be broadcast: "
                        f"barrier={barrier_values.shape}, vanilla={vanilla_values.shape}"
                    )
        
        # Broadcast breach mask to match value dimensions if needed
        mask = breach_mask
        while mask.dim() < barrier_values.dim():
            mask = mask.unsqueeze(-1)
        
        # Expand mask to match full shape
        mask = mask.expand_as(barrier_values)
        
        # Select: vanilla where breached, barrier where not breached
        result = torch.where(mask, vanilla_values, barrier_values)
        
        return result
    
    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Price up-and-in barrier option with path-specific barrier handling.
        
        Args:
            S: Spot price(s) [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Option price per path [M] or [M, ...]
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'price')
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Compute delta with path-specific barrier handling.
        
        CRITICAL: Paths that breached use vanilla delta, paths that haven't use barrier delta.
        
        Args:
            S: Spot price(s) [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Delta per path [M] or [M, ...]
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'delta')
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Compute gamma with path-specific barrier handling.
        
        Args:
            S: Spot price(s) [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Gamma per path [M] or [M, ...]
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'gamma')
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Compute vega with path-specific barrier handling.
        
        Args:
            S: Spot price(s) [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Vega per path [M] or [M, ...]
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'vega')
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Compute theta with path-specific barrier handling.
        
        Args:
            S: Spot price(s) [M] or [M, ...]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Theta per path [M] or [M, ...]
        """
        return self._compute_mixed_greek(S, K, step_idx, N, h0, 'theta')
    
    def compute_barrier_delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: Optional[float] = None) -> torch.Tensor:
        """
        Compute sensitivity of option price to barrier level (barrier delta).
        
        This requires autograd through the barrier_level parameter.
        Only applicable to paths that haven't breached (breached paths have zero sensitivity).
        
        Args:
            S: Spot price(s) [M]
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Variance (optional, scalar)
            
        Returns:
            Barrier delta per path [M]
        """
        # Get breach status
        breach_mask = self.check_and_update_barrier(S)
        
        # For breached paths, barrier delta is zero
        # For non-breached paths, compute via autograd
        
        if not torch.any(~breach_mask):
            # All paths breached - return zeros
            return torch.zeros_like(S)
        
        # Ensure barrier_level requires grad
        if not self.barrier_level.requires_grad:
            raise ValueError("barrier_level must have requires_grad=True for barrier_delta computation")
        
        # Compute price with autograd enabled
        # CRITICAL: Pass h0 as scalar for proper broadcasting
        if h0 is not None:
            if not isinstance(h0, torch.Tensor):
                h0_tensor = torch.tensor(h0, dtype=torch.float32, device=self.device, requires_grad=True)
            else:
                h0_tensor = h0
            prices = self.barrier_option.price(S, K, step_idx, N, h0_tensor)
        else:
            prices = self.barrier_option.price(S, K, step_idx, N)
        
        # Compute gradient of mean price w.r.t. barrier_level
        mean_price = prices.mean()
        
        grad = torch.autograd.grad(
            outputs=mean_price,
            inputs=self.barrier_level,
            create_graph=False,
            retain_graph=True
        )[0]
        
        # The gradient is a scalar - replicate for all non-breached paths
        barrier_deltas = torch.full_like(S, grad.item())
        
        # Zero out breached paths
        barrier_deltas = torch.where(breach_mask, torch.zeros_like(S), barrier_deltas)
        
        return barrier_deltas
