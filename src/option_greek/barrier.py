import torch
import torch.nn as nn
import math
import os
from .base import DerivativeBase
from typing import Dict

class BarrierOptionNN(nn.Module):
    """Neural network architecture for barrier option pricing."""
    def __init__(self, input_dim=33):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 4096); self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096);      self.bn2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2048);      self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 1024);      self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 512);       self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 256);        self.bn6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01))
        x = self.dropout(torch.nn.functional.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01))
        x = torch.nn.functional.leaky_relu(self.bn6(self.fc6(x)), negative_slope=0.01)
        x = self.fc7(x)
        return torch.clamp(x, min=0.0)


def create_features_batched(S0, K, T, r, barrier, h0, option_type="call"):
    """
    Batched feature engineering - processes all spot prices at once.
    
    Args:
        S0: [batch_size] tensor of spot prices
        K: scalar or [batch_size] tensor
        T: scalar or [batch_size] tensor
        r: scalar or [batch_size] tensor
        barrier: scalar or [batch_size] tensor
        h0: scalar or [batch_size] tensor
        option_type: 'call' or 'put'
    
    Returns:
        [batch_size, 33] feature tensor
    """
    # Ensure all inputs are tensors with same shape - use as_tensor to avoid copying
    if not isinstance(K, torch.Tensor):
        K = torch.full_like(S0, K)
    else:
        K = K.expand_as(S0) if K.numel() == 1 else K
    
    if not isinstance(T, torch.Tensor):
        T = torch.full_like(S0, T)
    else:
        T = T.expand_as(S0) if T.numel() == 1 else T
    
    if not isinstance(r, torch.Tensor):
        r = torch.full_like(S0, r)
    else:
        r = r.expand_as(S0) if r.numel() == 1 else r
    
    if not isinstance(barrier, torch.Tensor):
        barrier = torch.full_like(S0, barrier)
    else:
        barrier = barrier.expand_as(S0) if barrier.numel() == 1 else barrier
    
    if not isinstance(h0, torch.Tensor):
        h0 = torch.full_like(S0, h0)
    else:
        h0 = h0.expand_as(S0) if h0.numel() == 1 else h0
    
    # Compute all features in batched manner
    moneyness = S0 / K
    barrier_distance_S = barrier / S0
    barrier_distance_K = barrier / K
    log_m = torch.log(moneyness)
    log_HS = torch.log(barrier_distance_S)
    log_HK = torch.log(barrier_distance_K)
    sqrt_T = torch.sqrt(T)
    vol_proxy = torch.sqrt(h0)
    vol_time = vol_proxy * sqrt_T
    drift_adj = (r - 0.5 * h0) * T
    
    if option_type == "call":
        intrinsic = torch.clamp(S0 - K, min=0.0)
        is_itm = (S0 > K).float()
        is_otm = (S0 < K).float()
    else:
        intrinsic = torch.clamp(K - S0, min=0.0)
        is_itm = (S0 < K).float()
        is_otm = (S0 > K).float()
    
    dist_atm = torch.abs(log_m)
    is_atm = (torch.abs(moneyness - 1.0) < 0.05).float()
    
    breach_risk = torch.clamp(1.0 - (barrier - S0) / S0, min=0.0)
    safe_dist = (barrier - S0) / torch.clamp(vol_proxy * S0 * sqrt_T, min=1e-8)
    time_decay = 1.0 / (T + 1e-3)
    short_dated = (T < 0.1).float()
    
    variance_term = h0 * T
    vol_of_vol = h0 ** 1.5
    
    # Stack all features: [batch_size, 33]
    features = torch.stack([
        moneyness, K, T, r,
        barrier_distance_S, barrier_distance_K, h0,
        log_m, log_HS, log_HK,
        sqrt_T, vol_proxy, vol_time, drift_adj,
        S0, barrier, S0 * r, K * r, barrier * r,
        barrier - S0, barrier - K, S0 - K,
        intrinsic,
        dist_atm, is_itm, is_otm, is_atm,
        breach_risk, safe_dist,
        time_decay, short_dated,
        variance_term, vol_of_vol
    ], dim=-1)
    
    return features


class BarrierOption(DerivativeBase):
    """Barrier option using neural network surrogate model with batched inference."""
    
    def __init__(self, model_path: str, barrier_level: float, 
                 barrier_type: str = "up-and-in", option_type: str = "call",
                 r_annual: float = 0.04, device: str = "cpu"):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth)
            barrier_level: Barrier level (e.g., 120.0)
            barrier_type: Type of barrier ('up-and-in', 'up-and-out', etc.)
            option_type: 'call' or 'put'
            r_annual: Annual risk-free rate
            device: 'cpu' or 'cuda'
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Barrier model not found at {model_path}. "
                f"Please run: python scripts/download_models.py"
            )
        
        self.device = torch.device(device)
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.r_annual = r_annual
        self.r_daily = r_annual / 252.0
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = BarrierOptionNN(input_dim=33).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        # Disable dropout at inference
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        
        self.mean = checkpoint["mean"].to(self.device)
        self.std = checkpoint["std"].to(self.device)
        
        print(f"[INFO] Loaded barrier model from {model_path}")
        print(f"[INFO] Test MAE: {checkpoint.get('test_mae', 'N/A')}")
    
    def _compute_time_to_maturity(self, step_idx: int, N: int) -> float:
        """Convert step index to time to maturity in years."""
        days_remaining = N - step_idx
        return days_remaining / 252.0
    
    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """
        Compute barrier option price with batched inference.
        
        Args:
            S: Spot price(s) - tensor of any shape
            K: Strike price
            step_idx: Current step index (0 to N)
            N: Total steps (maturity in days)
            h0: Current variance
        
        Returns:
            Price tensor with same shape as S
        """
        T = self._compute_time_to_maturity(step_idx, N)
        
        # Convert inputs to tensors
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        # Create batched tensors for all inputs
        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)
        
        # Batched feature computation
        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch, 
            self.option_type
        )
        
        # Normalize features
        features_norm = (features - self.mean) / self.std
        
        # Single batched forward pass
        with torch.no_grad():
            prices = self.model(features_norm).squeeze(-1)
        
        return prices.reshape(original_shape)
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute delta via automatic differentiation with batching."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)
        
        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute gradients
        delta = torch.autograd.grad(prices.sum(), S_t, create_graph=False)[0]
        
        return delta.reshape(original_shape)
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute gamma via automatic differentiation with batching."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)
        
        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute second derivatives
        first_derivative = torch.autograd.grad(prices.sum(), S_t, create_graph=True)[0]
        gamma = torch.autograd.grad(first_derivative.sum(), S_t, create_graph=False)[0]
        
        return gamma.reshape(original_shape)
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute vega via automatic differentiation with batching."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device, requires_grad=True)
        
        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)
        
        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute vega for batch
        vega_raw = torch.autograd.grad(prices.mean(), h0_t, allow_unused=True)[0]
        if vega_raw is not None:
            vega = vega_raw * 2 * torch.sqrt(h0_t)
            return torch.full(original_shape, vega.item(), device=self.device)
        else:
            return torch.zeros(original_shape, device=self.device)
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute theta via automatic differentiation with batching."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device, requires_grad=True)
        
        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)
        
        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std
        prices = self.model(features_norm).squeeze(-1)
        
        # Compute theta for batch
        theta_raw = torch.autograd.grad(prices.mean(), T_t, allow_unused=True)[0]
        if theta_raw is not None:
            theta = -theta_raw
            return torch.full(original_shape, theta.item(), device=self.device)
        else:
            return torch.zeros(original_shape, device=self.device)
