"""
Abstract base class for derivative instruments.

This module defines the common interface that all derivative pricing
classes must implement, ensuring consistency across different derivative
types (vanilla, barrier, exotic, etc.).
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict


class DerivativeBase(ABC):
    """
    Abstract base class for all derivative instruments.
    
    This class defines the standard interface for pricing and computing
    Greeks for any derivative instrument. All concrete derivative classes
    (VanillaOption, BarrierOption, etc.) must inherit from this class
    and implement all abstract methods.
    
    Standard interface uses:
        S: Spot price (torch.Tensor)
        K: Strike price (float)
        step_idx: Current time step (int, 0 to N)
        N: Total maturity in days (int)
    """
    
    @abstractmethod
    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> torch.Tensor:
        """
        Calculate the option price.
        
        Args:
            S: Spot price(s) - can be scalar or tensor of any shape
            K: Strike price
            step_idx: Current time step index (0 = start, N = expiry)
            N: Total number of time steps (maturity in days)
        
        Returns:
            Option price tensor with the same shape as S
            
        Example:
            >>> price = derivative.price(S=100.0, K=100.0, step_idx=0, N=252)
        """
        pass
    
    @abstractmethod
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> torch.Tensor:
        """
        Calculate delta: ∂V/∂S (first derivative of value with respect to spot).
        
        Delta measures how much the option value changes when the spot price
        moves by $1. For a call: delta ∈ [0, 1], for a put: delta ∈ [-1, 0].
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current time step index
            N: Total number of time steps
        
        Returns:
            Delta tensor with the same shape as S
        """
        pass
    
    @abstractmethod
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> torch.Tensor:
        """
        Calculate gamma: ∂²V/∂S² (second derivative of value with respect to spot).
        
        Gamma measures the rate of change of delta. High gamma means delta
        is changing rapidly, which means more frequent rebalancing is needed
        for delta hedging. Gamma is always positive for long options.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current time step index
            N: Total number of time steps
        
        Returns:
            Gamma tensor with the same shape as S
        """
        pass
    
    @abstractmethod
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> torch.Tensor:
        """
        Calculate vega: ∂V/∂σ (derivative of value with respect to volatility).
        
        Vega measures how much the option value changes when volatility changes.
        Long options have positive vega (benefit from increased volatility).
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current time step index
            N: Total number of time steps
        
        Returns:
            Vega tensor with the same shape as S
        """
        pass
    
    @abstractmethod
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> torch.Tensor:
        """
        Calculate theta: -∂V/∂T (negative derivative of value with respect to time).
        
        Theta represents time decay - how much value the option loses per day
        as it approaches expiration. By convention, theta is typically negative
        for long option positions (you lose value as time passes).
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current time step index
            N: Total number of time steps
        
        Returns:
            Theta tensor with the same shape as S
        """
        pass
    
    def compute_all_greeks(self, S: torch.Tensor, K: float, step_idx: int, N: int) -> Dict[str, torch.Tensor]:
        """
        Convenience method to compute price and all Greeks at once.
        
        This is useful for hedging environments where you need all Greeks
        simultaneously for portfolio management.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current time step index
            N: Total number of time steps
        
        Returns:
            Dictionary containing:
                - 'price': Option price
                - 'delta': Delta (∂V/∂S)
                - 'gamma': Gamma (∂²V/∂S²)
                - 'vega': Vega (∂V/∂σ)
                - 'theta': Theta (-∂V/∂T)
                
        Example:
            >>> greeks = derivative.compute_all_greeks(S=100.0, K=100.0, step_idx=0, N=252)
            >>> print(f"Price: {greeks['price']}, Delta: {greeks['delta']}")
        """
        return {
            'price': self.price(S, K, step_idx, N),
            'delta': self.delta(S, K, step_idx, N),
            'gamma': self.gamma(S, K, step_idx, N),
            'vega': self.vega(S, K, step_idx, N),
            'theta': self.theta(S, K, step_idx, N)
        }
    
    def __repr__(self) -> str:
        """String representation of the derivative."""
        return f"{self.__class__.__name__}()"


class TimeDependentDerivative(DerivativeBase):
    """
    Base class for derivatives with discrete time steps (hedging environments).
    
    Provides helper methods for converting between step indices and calendar time.
    This is useful when you need to work with time-to-maturity in years rather
    than step indices.
    """
    
    def __init__(self, r_annual: float, days_per_year: int = 252):
        """
        Initialize time-dependent derivative.
        
        Args:
            r_annual: Annual risk-free rate (e.g., 0.04 for 4%)
            days_per_year: Trading days per year (default: 252)
        """
        self.r_annual = r_annual
        self.r_daily = r_annual / days_per_year
        self.days_per_year = days_per_year
    
    def time_to_maturity(self, step_idx: int, N: int) -> float:
        """
        Convert step index to time to maturity in years.
        
        Args:
            step_idx: Current step (0 to N)
            N: Total steps (maturity)
            
        Returns:
            Time to maturity in years
            
        Example:
            >>> # At start of 1-year option
            >>> self.time_to_maturity(step_idx=0, N=252)
            1.0
            
            >>> # Halfway through 1-year option
            >>> self.time_to_maturity(step_idx=126, N=252)
            0.5
            
            >>> # At expiration
            >>> self.time_to_maturity(step_idx=252, N=252)
            0.0
        """
        days_remaining = N - step_idx
        return days_remaining / self.days_per_year
    
    def steps_remaining(self, T: float) -> int:
        """
        Convert time to maturity (years) to number of steps remaining.
        
        Args:
            T: Time to maturity in years
            
        Returns:
            Number of steps (days) remaining
            
        Example:
            >>> # 1 year to maturity
            >>> self.steps_remaining(1.0)
            252
            
            >>> # 3 months to maturity
            >>> self.steps_remaining(0.25)
            63
        """
        return int(T * self.days_per_year)
    
    def __repr__(self) -> str:
        """String representation with rate info."""
        return f"{self.__class__.__name__}(r_annual={self.r_annual:.4f})"
