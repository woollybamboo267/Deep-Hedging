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
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = BarrierOptionNN(input_dim=33).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
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
        
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        
        if T < 1e-6:
            if self.option_type.lower() == "call":
                intrinsic = torch.clamp(S_t - K, min=0.0)
            else:
                intrinsic = torch.clamp(K - S_t, min=0.0)
            return intrinsic
        
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
        
        with torch.no_grad():
            prices = self.model(features_norm).squeeze(-1)
        
        return prices.reshape(original_shape)
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float, epsilon: float = 0.01) -> torch.Tensor:
        """
        Compute delta using finite difference method.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Current variance
            epsilon: Bump size for finite difference (default 0.01)
        
        Returns:
            Delta tensor
        """
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        
        S_up = S_t + epsilon
        S_down = S_t - epsilon
        
        price_up = self.price(S_up, K, step_idx, N, h0)
        price_down = self.price(S_down, K, step_idx, N, h0)
        
        delta = (price_up - price_down) / (2.0 * epsilon)
        
        return delta
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float, epsilon: float = 0.01) -> torch.Tensor:
        """
        Compute gamma using finite difference method.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Current variance
            epsilon: Bump size for finite difference (default 0.01)
        
        Returns:
            Gamma tensor
        """
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        
        S_up = S_t + epsilon
        S_down = S_t - epsilon
        
        price_center = self.price(S_t, K, step_idx, N, h0)
        price_up = self.price(S_up, K, step_idx, N, h0)
        price_down = self.price(S_down, K, step_idx, N, h0)
        
        gamma = (price_up - 2.0 * price_center + price_down) / (epsilon ** 2)
        
        return gamma
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Compute vega using finite difference method.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Current variance
            epsilon: Bump size for variance (default 1e-8)
        
        Returns:
            Vega tensor
        """
        h0_up = h0 + epsilon
        h0_down = h0 - epsilon
        
        price_up = self.price(S, K, step_idx, N, h0_up)
        price_down = self.price(S, K, step_idx, N, h0_down)
        
        vega_h0 = (price_up - price_down) / (2.0 * epsilon)
        
        vega = vega_h0 * 2.0 * torch.sqrt(torch.tensor(h0, device=self.device))
        
        return vega
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float, epsilon_days: int = 1) -> torch.Tensor:
        """
        Compute theta using finite difference method.
        
        Args:
            S: Spot price(s)
            K: Strike price
            step_idx: Current step index
            N: Total steps
            h0: Current variance
            epsilon_days: Time step in days (default 1)
        
        Returns:
            Theta tensor
        """
        T = self._compute_time_to_maturity(step_idx, N)
        
        if T < epsilon_days / 252.0:
            S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
            return torch.zeros_like(S_t)
        
        price_now = self.price(S, K, step_idx, N, h0)
        price_future = self.price(S, K, step_idx + epsilon_days, N, h0)
        
        theta = -(price_future - price_now) / epsilon_days
        
        return theta


# Test script remains the same
