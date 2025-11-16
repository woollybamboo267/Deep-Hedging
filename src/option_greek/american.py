import torch
import torch.nn as nn
import os
import math
import numpy as np
from .base import DerivativeBase
from typing import Dict, Union


# ---------------------- Neural Network ----------------------

class AmericanOptionNN(nn.Module):
    """Neural network architecture for American option pricing."""
    def __init__(self, input_dim=18):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048); self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048);      self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 1024);      self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512);       self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256);        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 128);        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn1(self.fc1(x)), 0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)), 0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn3(self.fc3(x)), 0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn4(self.fc4(x)), 0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn5(self.fc5(x)), 0.01))
        x = torch.nn.functional.leaky_relu(self.bn6(self.fc6(x)), 0.01)
        x = self.fc7(x)
        return torch.clamp(x, min=0.0)


# ---------------------- Feature Engineering ----------------------

def create_features_batched(S0, K, T, r, h0):
    """
    Feature creation for batched inference with proper autograd support.
    
    Creates 18 features for American PUT option pricing:
    1. moneyness (S/K)
    2. K: Strike level
    3. T: Time to maturity
    4. r: Risk-free rate
    5. h0: Initial variance
    6. log(S/K): Log moneyness
    7. sqrt(T): Time scaling
    8. vol * sqrt(T): Total volatility
    9. intrinsic: PUT intrinsic value
    10. tv_ratio: Time value ratio
    11. distance_to_boundary: Distance to exercise boundary
    12. boundary_proximity: Probability of crossing boundary
    13. interest_incentive: Rate incentive for early exercise
    14. vol_adj_moneyness: Volatility-adjusted moneyness
    15. d1_proxy: Black-Scholes d1
    16. is_atm: At-the-money indicator
    17. is_slight_itm: Slightly in-the-money indicator
    18. is_slight_otm: Slightly out-of-the-money indicator
    
    CRITICAL FIX: Handles both scalar and batched parameters correctly.
    When h0 or T are scalar tensors with requires_grad, they're properly
    broadcast to match batch size while maintaining gradient flow.
    """
    def ensure_tensor(x, reference_shape):
        """
        Convert parameter to tensor with proper shape for batching.
        
        Args:
            x: Parameter (can be float, tensor scalar, or batched tensor)
            reference_shape: Shape to match (typically S0.shape)
        """
        if isinstance(x, torch.Tensor):
            # Already a tensor
            if x.dim() == 0:
                # Scalar tensor - expand to batch size while keeping gradients
                return x.unsqueeze(0).expand(reference_shape)
            elif x.shape == reference_shape:
                # Already correct shape
                return x
            elif x.shape[0] == reference_shape[0]:
                # Correct batch size
                return x
            else:
                raise ValueError(f"Tensor shape {x.shape} incompatible with reference {reference_shape}")
        else:
            # Float or other type - convert to tensor
            return torch.full(reference_shape, x, dtype=S0.dtype, device=S0.device)

    # Ensure all parameters have consistent batch dimension
    K_t = ensure_tensor(K, S0.shape)
    T_t = ensure_tensor(T, S0.shape)
    r_t = ensure_tensor(r, S0.shape)
    h0_t = ensure_tensor(h0, S0.shape)
    
    # Compute basic features
    moneyness = S0 / K_t
    log_m = torch.log(moneyness)
    vol = torch.sqrt(h0_t)
    sqrt_T = torch.sqrt(T_t)
    vol_time = vol * sqrt_T
    
    # Intrinsic value for PUT
    intrinsic = torch.clamp(K_t - S0, min=0.0)
    
    # Time value ratio
    time_value_proxy = vol * sqrt_T * K_t * 0.4
    total_value_est = intrinsic + time_value_proxy
    tv_ratio = time_value_proxy / (total_value_est + 1e-8)
    
    # Early exercise boundary features
    S_critical = K_t / (1 + r_t * T_t + vol * sqrt_T)
    distance_to_boundary = (S0 - S_critical) / K_t
    boundary_proximity = torch.exp(-distance_to_boundary**2 / (2 * vol**2 * T_t + 1e-8))
    interest_incentive = r_t * T_t * (K_t / (S0 + 1e-8))
    
    # Advanced features
    vol_adj_moneyness = log_m / (vol * sqrt_T + 1e-8)
    d1_proxy = (log_m + (r_t + 0.5*h0_t)*T_t) / (vol * sqrt_T + 1e-8)
    
    # Moneyness indicators
    is_atm = (torch.abs(log_m) < 0.1).float()
    is_slight_itm = ((log_m > -0.2) & (log_m < -0.05)).float()
    is_slight_otm = ((log_m > 0.05) & (log_m < 0.2)).float()
    
    features = torch.stack([
        moneyness, K_t, T_t, r_t, h0_t,
        log_m, sqrt_T, vol_time,
        intrinsic, tv_ratio,
        distance_to_boundary, boundary_proximity, interest_incentive,
        vol_adj_moneyness, d1_proxy,
        is_atm, is_slight_itm, is_slight_otm
    ], dim=-1)

    return features


# ---------------------- American Option Wrapper ----------------------

class AmericanOption(DerivativeBase):
    """American PUT option using neural network surrogate model."""

    def __init__(self, model_path: str, option_type: str = "put",
                 r_annual: float = 0.04, device: str = "cpu"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"American model not found at {model_path}. "
                f"Please run: python scripts/download_models.py"
            )

        self.device = torch.device(device)
        self.option_type = option_type
        self.r_annual = r_annual
        self.r_daily = r_annual / 252.0

        if option_type != "put":
            raise ValueError("Only American PUT options are currently supported")

        # ---------------------- Fix for PyTorch 2.6 unpickling ----------------------
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = AmericanOptionNN(18).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        # Disable dropout for inference
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        self.mean = checkpoint["mean"].to(self.device)
        self.std = checkpoint["std"].to(self.device)

        print(f"[INFO] Loaded American model from {model_path}")
        print(f"[INFO] Test MAE: {checkpoint.get('test_mae', 'N/A')}")

    # ---------------------- Helpers ----------------------

    def _compute_time_to_maturity(self, step_idx: int, N: int) -> float:
        return (N - step_idx) / 252.0

    # ---------------------- Pricing ----------------------

    def price(self, S, K, step_idx, N, h0):
        """
        Price American PUT option.
        
        Args:
            S: Spot price(s) - tensor of shape [M] or scalar
            K: Strike price - float
            step_idx: Current time step - int
            N: Total time steps - int
            h0: Variance - float or scalar tensor (can have requires_grad=True)
        """
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        orig_shape = S_t.shape

        if T < 1e-6:
            # At expiry, return intrinsic value
            K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
            return torch.clamp(K_t - S_t, min=0.0)

        S_flat = S_t.reshape(-1)
        
        # Convert h0 to tensor if needed, preserving gradients
        if not isinstance(h0, torch.Tensor):
            h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        else:
            h0_t = h0
        
        features = create_features_batched(
            S_flat, K, T, self.r_daily * 252.0, h0_t
        )
        features_norm = (features - self.mean) / self.std
        with torch.no_grad():
            prices = self.model(features_norm).squeeze(-1)
        return prices.reshape(orig_shape)

    # ---------------------- Greeks ----------------------

    def delta(self, S, K, step_idx, N, h0):
        """Delta for American PUT option."""
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        orig_shape = S_t.shape

        if T < 1e-6:
            # At expiry: delta = -1 if ITM, 0 if OTM
            K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
            return -(S_t < K_t).float()

        S_t = S_t.clone().detach().requires_grad_(True)
        S_flat = S_t.reshape(-1)
        
        # Convert h0 to tensor if needed
        if not isinstance(h0, torch.Tensor):
            h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        else:
            h0_t = h0
        
        features = create_features_batched(
            S_flat, K, T, self.r_daily * 252.0, h0_t
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        delta = torch.autograd.grad(prices.sum(), S_t, create_graph=False)[0]
        return delta.reshape(orig_shape)

    def gamma(self, S, K, step_idx, N, h0):
        """Compute gamma via second derivative."""
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        orig_shape = S_t.shape

        if T < 1e-6:
            return torch.zeros_like(S_t)

        S_t = S_t.clone().detach().requires_grad_(True)
        S_flat = S_t.reshape(-1)
        
        # Convert h0 to tensor if needed
        if not isinstance(h0, torch.Tensor):
            h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        else:
            h0_t = h0
        
        features = create_features_batched(
            S_flat, K, T, self.r_daily * 252.0, h0_t
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        first_derivative = torch.autograd.grad(prices.sum(), S_t, create_graph=True)[0]
        gamma = torch.autograd.grad(first_derivative.sum(), S_t, create_graph=False)[0]
        return gamma.reshape(orig_shape)

    def vega(self, S, K, step_idx, N, h0):
        """
        Compute vega with proper gradient tracking for h0.
        
        CRITICAL FIX: h0 is passed as scalar tensor and broadcast internally.
        """
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        orig_shape = S_t.shape
    
        if T < 1e-6:
            return torch.zeros_like(S_t)
    
        S_flat = S_t.reshape(-1)
        
        # Convert h0 to scalar tensor with gradient tracking
        if not isinstance(h0, torch.Tensor):
            h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            if not h0.requires_grad:
                h0_t = h0.clone().detach().requires_grad_(True)
            else:
                h0_t = h0
        
        # create_features_batched will handle broadcasting h0_t to match S_flat
        features = create_features_batched(
            S_flat, K, T, self.r_daily * 252.0, h0_t
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute gradient with respect to h0
        mean_price = prices.mean()
        vega_raw = torch.autograd.grad(mean_price, h0_t, allow_unused=True)[0]
        
        if vega_raw is None:
            return torch.zeros(orig_shape, device=self.device)
        
        # Convert d(price)/d(h0) to d(price)/d(sigma)
        # Since h0 = sigma^2, we have d/d(sigma) = d/d(h0) * 2*sigma
        sigma = torch.sqrt(h0_t)
        vega = vega_raw * 2 * sigma
        
        # Replicate scalar gradient across all paths
        return torch.full(orig_shape, vega.item(), device=self.device)
    

    def theta(self, S, K, step_idx, N, h0):
        """
        Compute theta with proper gradient tracking for T.
        
        CRITICAL FIX: T is passed as scalar tensor and broadcast internally.
        """
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        orig_shape = S_t.shape
    
        if T < 1e-6:
            return torch.zeros_like(S_t)
    
        S_flat = S_t.reshape(-1)
        
        # Create T as scalar tensor with gradient tracking
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Convert h0 to tensor if needed
        if not isinstance(h0, torch.Tensor):
            h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        else:
            h0_t = h0
        
        # create_features_batched will handle broadcasting T_t to match S_flat
        features = create_features_batched(
            S_flat, K, T_t, self.r_daily * 252.0, h0_t
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute gradient with respect to T
        mean_price = prices.mean()
        theta_raw = torch.autograd.grad(mean_price, T_t, allow_unused=True)[0]
        
        if theta_raw is None:
            return torch.zeros(orig_shape, device=self.device)
        
        # Negative because theta is -dV/dT (value decays with time)
        theta = -theta_raw
        
        # Replicate scalar gradient across all paths
        return torch.full(orig_shape, theta.item(), device=self.device)
