from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class DerivativeBase(ABC):
    """Abstract base class for all derivative instruments."""
    
    @abstractmethod
    def price(self, S: torch.Tensor, K: float, step_idx: int, **kwargs) -> torch.Tensor:
        """Calculate option price."""
        pass
    
    @abstractmethod
    def delta(self, S: torch.Tensor, K: float, step_idx: int, **kwargs) -> torch.Tensor:
        """Calculate delta (∂V/∂S)."""
        pass
    
    @abstractmethod
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, **kwargs) -> torch.Tensor:
        """Calculate gamma (∂²V/∂S²)."""
        pass
    
    @abstractmethod
    def vega(self, S: torch.Tensor, K: float, step_idx: int, **kwargs) -> torch.Tensor:
        """Calculate vega (∂V/∂σ)."""
        pass
    
    @abstractmethod
    def theta(self, S: torch.Tensor, K: float, step_idx: int, **kwargs) -> torch.Tensor:
        """Calculate theta (-∂V/∂T)."""
        pass
