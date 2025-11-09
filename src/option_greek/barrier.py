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


def create_features(S0, K, T, r, barrier, h0, option_type="call"):
    """Feature engineering - must match training."""
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
    intrinsic = torch.clamp(S0 - K, min=0.0) if option_type == "call" else torch.clamp(K - S0, min=0.0)
    dist_atm = torch.abs(log_m)

    is_itm = torch.where(
        (S0 > K) if option_type == "call" else (S0 < K),
        torch.tensor(1.0, device=S0.device, dtype=S0.dtype),
        torch.tensor(0.0, device=S0.device, dtype=S0.dtype)
    )
    is_otm = torch.where(
        (S0 < K) if option_type == "call" else (S0 > K),
        torch.tensor(1.0, device=S0.device, dtype=S0.dtype),
        torch.tensor(0.0, device=S0.device, dtype=S0.dtype)
    )
    is_atm = torch.where(
        torch.abs(moneyness - 1.0) < 0.05,
        torch.tensor(1.0, device=S0.device, dtype=S0.dtype),
        torch.tensor(0.0, device=S0.device, dtype=S0.dtype)
    )

    breach_risk = torch.clamp(1.0 - (barrier - S0) / S0, min=0.0)
    safe_dist = (barrier - S0) / torch.clamp(vol_proxy * S0 * sqrt_T, min=1e-8)
    time_decay = 1.0 / (T + 1e-3)
    short_dated = torch.where(
        T < 0.1,
        torch.tensor(1.0, device=T.device, dtype=T.dtype),
        torch.tensor(0.0, device=T.device, dtype=T.dtype)
    )

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
    """Barrier option using neural network surrogate model."""
    
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
        Compute barrier option price.
        
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
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(self.r_daily * 252.0, dtype=torch.float32, device=self.device)  # Annual rate
        barrier_t = torch.tensor(self.barrier_level, dtype=torch.float32, device=self.device)
        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        
        # Handle batching
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)
        
        prices = []
        for s in S_flat:
            features = create_features(s, K_t, T_t, r_t, barrier_t, h0_t, self.option_type)
            features_norm = (features - self.mean) / self.std
            with torch.no_grad():
                price = self.model(features_norm.unsqueeze(0)).squeeze()
            prices.append(price)
        
        prices_tensor = torch.stack(prices).reshape(original_shape)
        return prices_tensor
    
    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute delta via automatic differentiation."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.tensor(S, dtype=torch.float32, requires_grad=True, device=self.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(self.r_daily * 252.0, dtype=torch.float32, device=self.device)
        barrier_t = torch.tensor(self.barrier_level, dtype=torch.float32, device=self.device)
        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        
        features = create_features(S_t, K_t, T_t, r_t, barrier_t, h0_t, self.option_type)
        features_norm = (features - self.mean) / self.std
        price = self.model(features_norm.unsqueeze(0)).squeeze()
        
        delta = torch.autograd.grad(price, S_t, create_graph=False)[0]
        return delta
    
    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute gamma via automatic differentiation."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.tensor(S, dtype=torch.float32, requires_grad=True, device=self.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(self.r_daily * 252.0, dtype=torch.float32, device=self.device)
        barrier_t = torch.tensor(self.barrier_level, dtype=torch.float32, device=self.device)
        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        
        features = create_features(S_t, K_t, T_t, r_t, barrier_t, h0_t, self.option_type)
        features_norm = (features - self.mean) / self.std
        price = self.model(features_norm.unsqueeze(0)).squeeze()
        
        delta = torch.autograd.grad(price, S_t, create_graph=True, retain_graph=True)[0]
        gamma = torch.autograd.grad(delta, S_t)[0]
        return gamma
    
    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute vega via automatic differentiation."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.tensor(S, dtype=torch.float32, device=self.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(self.r_daily * 252.0, dtype=torch.float32, device=self.device)
        barrier_t = torch.tensor(self.barrier_level, dtype=torch.float32, device=self.device)
        h0_t = torch.tensor(h0, dtype=torch.float32, requires_grad=True, device=self.device)
        
        features = create_features(S_t, K_t, T_t, r_t, barrier_t, h0_t, self.option_type)
        features_norm = (features - self.mean) / self.std
        price = self.model(features_norm.unsqueeze(0)).squeeze()
        
        vega_raw = torch.autograd.grad(price, h0_t, allow_unused=True)[0]
        vega = vega_raw * 2 * torch.sqrt(h0_t) if vega_raw is not None else torch.tensor(0.0, device=self.device)
        return vega
    
    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        """Compute theta via automatic differentiation."""
        T = self._compute_time_to_maturity(step_idx, N)
        
        S_t = torch.tensor(S, dtype=torch.float32, device=self.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.float32, requires_grad=True, device=self.device)
        r_t = torch.tensor(self.r_daily * 252.0, dtype=torch.float32, device=self.device)
        barrier_t = torch.tensor(self.barrier_level, dtype=torch.float32, device=self.device)
        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device)
        
        features = create_features(S_t, K_t, T_t, r_t, barrier_t, h0_t, self.option_type)
        features_norm = (features - self.mean) / self.std
        price = self.model(features_norm.unsqueeze(0)).squeeze()
        
        theta_raw = torch.autograd.grad(price, T_t, allow_unused=True)[0]
        theta = -theta_raw if theta_raw is not None else torch.tensor(0.0, device=self.device)
        return theta
