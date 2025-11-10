import torch
import torch.nn as nn
import os
from .base import DerivativeBase

class BarrierOptionNN(nn.Module):
    """Neural network architecture for barrier option pricing."""
    def __init__(self, input_dim=33):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
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
        K, T, r, barrier, h0: scalars or tensors broadcastable to S0 shape
        option_type: 'call' or 'put'
    Returns:
        [batch_size, 33] feature tensor
    """
    # Make sure all inputs are tensors and broadcastable
    K = torch.as_tensor(K, device=S0.device).expand_as(S0)
    T = torch.as_tensor(T, device=S0.device).expand_as(S0)
    r = torch.as_tensor(r, device=S0.device).expand_as(S0)
    barrier = torch.as_tensor(barrier, device=S0.device).expand_as(S0)
    h0 = torch.as_tensor(h0, device=S0.device).expand_as(S0)

    moneyness = S0 / K
    barrier_distance_S = barrier / S0
    barrier_distance_K = barrier / K
    log_m = torch.log(moneyness + 1e-8)
    log_HS = torch.log(barrier_distance_S + 1e-8)
    log_HK = torch.log(barrier_distance_K + 1e-8)
    sqrt_T = torch.sqrt(T + 1e-8)
    vol_proxy = torch.sqrt(h0 + 1e-8)
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

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = BarrierOptionNN(input_dim=33).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        # Disable dropout at inference
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        self.mean = checkpoint["mean"].to(self.device)
        self.std = checkpoint["std"].to(self.device)

    def _compute_time_to_maturity(self, step_idx: int, N: int) -> float:
        days_remaining = N - step_idx
        return days_remaining / 252.0

    def price(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
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

        with torch.no_grad():
            prices = self.model(features_norm).squeeze(-1)

        return prices.reshape(original_shape)

    def delta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device, requires_grad=True)
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
        delta = torch.autograd.grad(prices.sum(), S_t, create_graph=False)[0]

        return delta.reshape(original_shape)

    def gamma(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device, requires_grad=True)
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
        first_derivative = torch.autograd.grad(prices.sum(), S_t, create_graph=True)[0]
        gamma = torch.autograd.grad(first_derivative.sum(), S_t, create_graph=False)[0]

        return gamma.reshape(original_shape)

    def vega(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        T = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)

        h0_t = torch.tensor(h0, dtype=torch.float32, device=self.device, requires_grad=True)

        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0_t)

        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std

        prices = self.model(features_norm).squeeze(-1)
        vega_raw = torch.autograd.grad(prices.mean(), h0_t, allow_unused=True)[0]

        if vega_raw is not None:
            vega = vega_raw * 2 * torch.sqrt(h0_t)
            return torch.full(original_shape, vega.item(), device=self.device)
        else:
            return torch.zeros(original_shape, device=self.device)

    def theta(self, S: torch.Tensor, K: float, step_idx: int, N: int, h0: float) -> torch.Tensor:
        T_val = self._compute_time_to_maturity(step_idx, N)
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        original_shape = S_t.shape
        S_flat = S_t.reshape(-1)

        T_t = torch.tensor(T_val, dtype=torch.float32, device=self.device, requires_grad=True)

        K_batch = torch.full_like(S_flat, K)
        T_batch = torch.full_like(S_flat, T_t)
        r_batch = torch.full_like(S_flat, self.r_daily * 252.0)
        barrier_batch = torch.full_like(S_flat, self.barrier_level)
        h0_batch = torch.full_like(S_flat, h0)

        features = create_features_batched(
            S_flat, K_batch, T_batch, r_batch, barrier_batch, h0_batch,
            self.option_type
        )
        features_norm = (features - self.mean) / self.std

        prices = self.model(features_norm).squeeze(-1)
        theta_raw = torch.autograd.grad(prices.mean(), T_t, allow_unused=True)[0]

        if theta_raw is not None:
            theta = -theta_raw
            return torch.full(original_shape, theta.item(), device=self.device)
        else:
            return torch.zeros(original_shape, device=self.device)
