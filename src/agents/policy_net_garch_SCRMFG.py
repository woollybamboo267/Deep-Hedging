# Clean, reformatted, and corrected version of the provided code.
# Purpose: Policy gradient LSTM + GARCH hedging environment with
#          static/floating option grids and ledger tracking.
#
# Notes:
# - Imports of repo-internal option classes are done lazily inside methods
#   to avoid circular import problems that may exist in the original project.
# - I fixed obvious syntax/formatting errors (missing newlines, wrong indices,
#   typos like "RetrySContinue0", and consistent use of num_layers).
# - The semantics/structure are preserved, but this version is easier to read
#   and should be syntactically valid Python.
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ----------------------------------------
# RMSNorm and symexp activation
# ----------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential: symexp(x) = sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class ResidualLSTMBlock(nn.Module):
    """Residual LSTM block: x → RMSNorm → LSTMCell → (+residual) → out"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.rms_norm = RMSNorm(hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        x_norm = self.rms_norm(x)
        h_new, c_new = self.lstm_cell(x_norm, hidden_state)
        out = x + h_new  # Residual connection
        return out, (h_new, c_new)


# ----------------------------------------
# Mueller et al. (2024) Policy Network
# ----------------------------------------
class PolicyNetGARCH(nn.Module):
    """
    Mueller et al. (2024) exact architecture for deep hedging.
    
    Architecture:
      Input → Linear → [4x Residual LSTM Blocks] → Linear → symexp → clamp
    
    Key specifications:
    - LSTM hidden size: 32 (not 128 or 256!)
    - 4 residually stacked LSTM blocks with RMSNorm
    - Output activation: symexp (not tanh!)
    - Final layer initialization: He init × 1e-3, zero bias
    """

    def __init__(
        self,
        obs_dim: int = 5,
        hidden_size: int = 32,  # Mueller uses 32!
        n_hedging_instruments: int = 2,
        num_lstm_blocks: int = 4,  # 4 residual blocks
        use_action_recurrence: bool = True,
        max_option_contracts: float = 10.0,
    ):
        super().__init__()
        self.n_hedging_instruments = n_hedging_instruments
        self.hidden_size = hidden_size
        self.use_action_recurrence = use_action_recurrence
        self.max_option_contracts = max_option_contracts
        
        # Input size includes previous actions if enabled
        input_size = obs_dim + n_hedging_instruments if use_action_recurrence else obs_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 4 Residual LSTM blocks
        self.lstm_blocks = nn.ModuleList([
            ResidualLSTMBlock(hidden_size) for _ in range(num_lstm_blocks)
        ])
        
        # Output heads: one per hedging instrument
        self.instrument_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_hedging_instruments)
        ])
        
        # Initialize parameters (Mueller's way)
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize following Mueller et al. specifications"""
        # Input projection: Xavier
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Output heads: He init scaled by 1e-3, zero bias (CRITICAL!)
        for head in self.instrument_heads:
            nn.init.kaiming_uniform_(head.weight, a=0, mode='fan_in', nonlinearity='linear')
            head.weight.data *= 1e-3  # Scale down by 1000x
            nn.init.zeros_(head.bias)
    
    def forward(
        self, 
        obs_sequence: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            obs_sequence: [batch, seq_len, obs_dim]
            prev_actions: [batch, seq_len, n_instruments]
            hidden_states: List of (h, c) for each LSTM block
        
        Returns:
            outputs: List of [batch, seq_len] per instrument
            new_hidden_states: List of (h, c) per block
        """
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Concatenate observations with previous actions
        if self.use_action_recurrence and prev_actions is not None:
            x = torch.cat([obs_sequence, prev_actions], dim=-1)
        else:
            x = obs_sequence
        
        # Input projection
        x = self.input_proj(x)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device)
                )
                for _ in range(len(self.lstm_blocks))
            ]
        
        # Process through residual LSTM blocks sequentially over time
        outputs_per_timestep = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, hidden_size]
            
            # Pass through all 4 residual LSTM blocks
            block_hiddens = []
            for block_idx, lstm_block in enumerate(self.lstm_blocks):
                x_t, new_h = lstm_block(x_t, hidden_states[block_idx])
                block_hiddens.append(new_h)
            
            # Update hidden states for next timestep
            hidden_states = block_hiddens
            outputs_per_timestep.append(x_t)
        
        # Stack timesteps: [batch, seq_len, hidden_size]
        x = torch.stack(outputs_per_timestep, dim=1)
        
        # Generate outputs for each instrument
        outputs = []
        for i, head in enumerate(self.instrument_heads):
            out = head(x).squeeze(-1)  # [batch, seq_len]
            
            # Apply symexp activation
            out = symexp(out)
            
            # Apply position limits
            if i == 0:
                # Stock: reasonable hedge ratios
                out = torch.clamp(out, min=-5.0, max=5.0)
            else:
                # Options: hard limit on contracts
                out = torch.clamp(out, min=-self.max_option_contracts, max=self.max_option_contracts)
            
            outputs.append(out)
        
        return outputs, hidden_states
# ----------------------------------------
# Floating grid manager
# ----------------------------------------
class FloatingGridManager:
    def __init__(self, config: Dict[str, Any], derivative_class, sim_params: Dict[str, Any], 
                 precomputation_manager, garch_params: Dict[str, float]):  # ADD THESE
        grid_config = config["instruments"]["floating_grid"]
        self.enabled = grid_config.get("enabled", False)
        if not self.enabled:
            return

        self.moneyness_levels: List[float] = grid_config["moneyness_levels"]
        self.maturity_days: List[int] = grid_config["maturity_days"]
        self.option_type: str = grid_config.get("option_type", "call")
        self.derivative_class = derivative_class
        self.sim_params = sim_params
        
        # NEW: Store these for creating VanillaOptions
        self.precomputation_manager = precomputation_manager
        self.garch_params = garch_params

        self.grid_size = len(self.moneyness_levels) * len(self.maturity_days)
        expected_size = 1 + self.grid_size
        actual_size = config["instruments"]["n_hedging_instruments"]
        if expected_size != actual_size:
            raise ValueError(
                f"Grid size mismatch: expected {expected_size} (1 stock + {self.grid_size} options), got {actual_size}"
            )


    def create_derivative(self, S_current: float, bucket_idx: int, current_step: int, total_steps: int):
        moneyness, maturity_days = self.get_bucket_params(bucket_idx)
        K = moneyness * S_current
   
        deriv = self.derivative_class(
            precomputation_manager=self.precomputation_manager,
            garch_params=self.garch_params,
            option_type=self.option_type
        )
        
        deriv.K = K
        deriv.N = maturity_days
        deriv.created_at_step = current_step  # ← ADD THIS LINE

        return deriv
    def get_bucket_params(self, bucket_idx: int) -> Tuple[float, int]:
        """
        Convert flat bucket index to (moneyness, maturity_days).
        Bucket ordering: maturity varies fastest.
        """
        n_maturities = len(self.maturity_days)
        moneyness_idx = bucket_idx // n_maturities
        maturity_idx = bucket_idx % n_maturities
        return self.moneyness_levels[moneyness_idx], self.maturity_days[maturity_idx]

# ----------------------------------------
# Position ledger for floating grid mode
# ----------------------------------------
class VectorizedPositionLedger:
    """Fully vectorized position ledger - stores all positions as tensors."""
    
    def __init__(self, M: int, device: torch.device):
        self.M = M
        self.device = device
        
        # All positions stored as 1D tensors (will grow dynamically)
        self.path_indices = torch.empty(0, dtype=torch.long, device=device)
        self.quantities = torch.empty(0, dtype=torch.float32, device=device)
        self.strikes = torch.empty(0, dtype=torch.float32, device=device)
        self.expiry_steps = torch.empty(0, dtype=torch.long, device=device)
        self.created_at_steps = torch.empty(0, dtype=torch.long, device=device)
        self.original_maturities = torch.empty(0, dtype=torch.long, device=device)
        self.bucket_indices = torch.empty(0, dtype=torch.long, device=device)
        self.opening_spots = torch.empty(0, dtype=torch.float32, device=device)
        
        # Store single derivative template (all positions use same pricing logic)
        self.derivative_template = None
        
    def add_positions_batch(
        self,
        path_indices: torch.Tensor,  # [N] which paths
        quantities: torch.Tensor,     # [N] quantities
        strikes: torch.Tensor,        # [N] strikes
        expiry_steps: torch.Tensor,   # [N] expiry steps
        created_at_step: int,
        original_maturity: int,
        bucket_idx: int,
        opening_spots: torch.Tensor,  # [N] spots
        derivative_template
    ):
        """Add multiple positions at once - NO LOOP."""
        if self.derivative_template is None:
            self.derivative_template = derivative_template
        
        n_new = len(path_indices)
        
        # Single concatenation instead of loop
        self.path_indices = torch.cat([self.path_indices, path_indices])
        self.quantities = torch.cat([self.quantities, quantities])
        self.strikes = torch.cat([self.strikes, strikes])
        self.expiry_steps = torch.cat([self.expiry_steps, expiry_steps])
        self.created_at_steps = torch.cat([
            self.created_at_steps, 
            torch.full((n_new,), created_at_step, dtype=torch.long, device=self.device)
        ])
        self.original_maturities = torch.cat([
            self.original_maturities,
            torch.full((n_new,), original_maturity, dtype=torch.long, device=self.device)
        ])
        self.bucket_indices = torch.cat([
            self.bucket_indices,
            torch.full((n_new,), bucket_idx, dtype=torch.long, device=self.device)
        ])
        self.opening_spots = torch.cat([self.opening_spots, opening_spots])
    def remove_expired(self, current_step: int):
        """Remove positions whose expiry_step <= current_step."""
        active_mask = self.expiry_steps > current_step
        self.path_indices = self.path_indices[active_mask]
        self.quantities = self.quantities[active_mask]
        self.strikes = self.strikes[active_mask]
        self.expiry_steps = self.expiry_steps[active_mask]
        self.created_at_steps = self.created_at_steps[active_mask]
        self.original_maturities = self.original_maturities[active_mask]
        self.bucket_indices = self.bucket_indices[active_mask]
        self.opening_spots = self.opening_spots[active_mask]
    
    def compute_and_remove_expired(
        self, 
        S_current: torch.Tensor, 
        current_step: int, 
        h0: float
    ) -> torch.Tensor:
        """Compute payoffs for expired positions and remove them. VECTORIZED."""
        payoffs = torch.zeros(self.M, device=self.device)
        
        if len(self.expiry_steps) == 0:
            return payoffs
        
        # Find expired positions
        expired_mask = self.expiry_steps <= current_step
        
        if expired_mask.sum() == 0:
            return payoffs
        
        # Get data for expired positions
        expired_paths = self.path_indices[expired_mask]
        expired_quantities = self.quantities[expired_mask]
        expired_strikes = self.strikes[expired_mask]
        
        # Get spot prices for expired positions (vectorized indexing)
        expired_spots = S_current[expired_paths]
        
        # Compute payoffs (vectorized)
        opt_type = getattr(self.derivative_template, "option_type", "call").lower()
        if opt_type == "call":
            intrinsic_values = torch.clamp(expired_spots - expired_strikes, min=0.0)
        else:
            intrinsic_values = torch.clamp(expired_strikes - expired_spots, min=0.0)
        
        position_payoffs = expired_quantities * intrinsic_values
        
        # Aggregate payoffs per path using scatter_add
        payoffs.scatter_add_(0, expired_paths, position_payoffs)
        
        # Remove expired positions
        self.path_indices = self.path_indices[~expired_mask]
        self.quantities = self.quantities[~expired_mask]
        self.strikes = self.strikes[~expired_mask]
        self.expiry_steps = self.expiry_steps[~expired_mask]
        self.created_at_steps = self.created_at_steps[~expired_mask]
        self.original_maturities = self.original_maturities[~expired_mask]
        self.bucket_indices = self.bucket_indices[~expired_mask]
        self.opening_spots = self.opening_spots[~expired_mask]
        
        return payoffs
    
    def compute_total_value(
        self, 
        S_current: torch.Tensor, 
        current_step: int, 
        h0: float
    ) -> torch.Tensor:
        """Price all active positions. FULLY VECTORIZED."""
        portfolio_value = torch.zeros(self.M, device=self.device)
        
        if len(self.expiry_steps) == 0 or self.derivative_template is None:
            return portfolio_value
        
        # Filter active positions
        active_mask = self.expiry_steps > current_step
        if active_mask.sum() == 0:
            return portfolio_value
        
        active_paths = self.path_indices[active_mask]
        active_quantities = self.quantities[active_mask]
        active_strikes = self.strikes[active_mask]
        active_created = self.created_at_steps[active_mask]
        active_maturities = self.original_maturities[active_mask]
        
        # Get spot prices for active positions (vectorized)
        active_spots = S_current[active_paths]
        
        # Compute step indices
        step_indices = current_step - active_created
        
        # Validate indices
        valid_mask = (step_indices >= 0) & (step_indices < active_maturities)
        if valid_mask.sum() == 0:
            return portfolio_value
        
        # Filter to valid positions
        valid_spots = active_spots[valid_mask]
        valid_strikes = active_strikes[valid_mask]
        valid_steps = step_indices[valid_mask]
        valid_maturities = active_maturities[valid_mask]
        valid_quantities = active_quantities[valid_mask]
        valid_paths = active_paths[valid_mask]
        
        # BATCH PRICE ALL POSITIONS AT ONCE
        # Note: This assumes derivative.price() can handle batched inputs
        # If your derivative class doesn't support this, you'll need to add that capability
        prices = self._batch_price_positions(
            valid_spots, valid_strikes, valid_steps, valid_maturities, h0
        )
        
        position_values = valid_quantities * prices
        
        # Aggregate per path using scatter_add
        portfolio_value.scatter_add_(0, valid_paths, position_values)
        
        return portfolio_value
    
    def _batch_price_positions(
        self, 
        S: torch.Tensor, 
        K: torch.Tensor, 
        step_indices: torch.Tensor, 
        maturities: torch.Tensor, 
        h0: float
    ) -> torch.Tensor:
        """
        Batch price multiple positions with different parameters.
        This is the critical optimization - prices all positions in one call.
        """
        if self.derivative_template is None:
            return torch.zeros_like(S)
        
        # Group by (step_idx, maturity) to minimize pricing calls
        unique_configs = torch.stack([step_indices, maturities], dim=1)
        unique_keys, inverse_indices = torch.unique(unique_configs, dim=0, return_inverse=True)
        
        n_positions = len(S)
        prices = torch.zeros(n_positions, device=self.device)
        
        # Price each unique configuration
        for i in range(len(unique_keys)):
            mask = inverse_indices == i
            if mask.sum() == 0:
                continue
            
            step_idx = int(unique_keys[i, 0].item())
            maturity = int(unique_keys[i, 1].item())
            
            S_batch = S[mask]
            K_batch = K[mask]
            
            # Call derivative pricing (should support batched S and K)
            batch_prices = self.derivative_template.price(
                S=S_batch,
                K=K_batch,
                step_idx=step_idx,
                N=maturity,
                h0=h0
            )
            
            prices[mask] = batch_prices
        
        return prices
    
    def compute_greeks(
        self, 
        S_current: torch.Tensor, 
        current_step: int, 
        h0: float, 
        greek_name: str
    ) -> torch.Tensor:
        """Aggregate Greeks across all active positions. VECTORIZED."""
        greeks = torch.zeros(self.M, device=self.device)
        
        if len(self.expiry_steps) == 0 or self.derivative_template is None:
            return greeks
        
        active_mask = self.expiry_steps > current_step
        if active_mask.sum() == 0:
            return greeks
        
        active_paths = self.path_indices[active_mask]
        active_quantities = self.quantities[active_mask]
        active_strikes = self.strikes[active_mask]
        active_created = self.created_at_steps[active_mask]
        active_maturities = self.original_maturities[active_mask]
        
        active_spots = S_current[active_paths]
        step_indices = current_step - active_created
        
        valid_mask = (step_indices >= 0) & (step_indices < active_maturities)
        if valid_mask.sum() == 0:
            return greeks
        
        valid_spots = active_spots[valid_mask]
        valid_strikes = active_strikes[valid_mask]
        valid_steps = step_indices[valid_mask]
        valid_maturities = active_maturities[valid_mask]
        valid_quantities = active_quantities[valid_mask]
        valid_paths = active_paths[valid_mask]
        
        # Batch compute Greeks (similar to pricing)
        greek_values = self._batch_compute_greeks(
            valid_spots, valid_strikes, valid_steps, valid_maturities, h0, greek_name
        )
        
        position_greeks = valid_quantities * greek_values
        greeks.scatter_add_(0, valid_paths, position_greeks)
        
        return greeks
    
    def _batch_compute_greeks(
        self, 
        S: torch.Tensor, 
        K: torch.Tensor, 
        step_indices: torch.Tensor, 
        maturities: torch.Tensor, 
        h0: float,
        greek_name: str
    ) -> torch.Tensor:
        """Batch compute Greeks for multiple positions."""
        if self.derivative_template is None:
            return torch.zeros_like(S)
        
        unique_configs = torch.stack([step_indices, maturities], dim=1)
        unique_keys, inverse_indices = torch.unique(unique_configs, dim=0, return_inverse=True)
        
        n_positions = len(S)
        greeks = torch.zeros(n_positions, device=self.device)
        
        greek_method = getattr(self.derivative_template, greek_name)
        
        for i in range(len(unique_keys)):
            mask = inverse_indices == i
            if mask.sum() == 0:
                continue
            
            step_idx = int(unique_keys[i, 0].item())
            maturity = int(unique_keys[i, 1].item())
            
            S_batch = S[mask]
            K_batch = K[mask]
            
            batch_greeks = greek_method(
                S=S_batch,
                K=K_batch,
                step_idx=step_idx,
                N=maturity,
                h0=h0
            )
            
            greeks[mask] = batch_greeks
        
        return greeks
    
    def get_ledger_size(self) -> torch.Tensor:
        """Return number of active positions per path. VECTORIZED."""
        sizes = torch.zeros(self.M, dtype=torch.float32, device=self.device)
        if len(self.path_indices) == 0:
            return sizes
        
        # Count positions per path
        sizes.scatter_add_(
            0, 
            self.path_indices, 
            torch.ones_like(self.path_indices, dtype=torch.float32)
        )
        return sizes


# ----------------------------------------
# Hedging environment with GARCH dynamics
# ----------------------------------------
class HedgingEnvGARCH:
    """
    Environment that simulates a hedging problem under GARCH dynamics and supports:
      - Static hedging instruments (stock + provided instruments)
      - Floating grid hedging: stock + dynamic option buckets (ledger-managed)

    Important fields:
      - sim: simulation spec that contains S0, K, N, M, sigma, r, T, etc.
      - derivative: the hedged derivative instance
      - hedging_derivatives: list of derivative objects for static mode (None for floating)
    """

    def __init__(
        self,
        sim,
        derivative,
        hedging_derivatives: Optional[List] = None,
        garch_params: Optional[Dict[str, Any]] = None,
        n_hedging_instruments: int = 2,
        dt_min: float = 1e-10,
        device: str = "cpu",
        transaction_costs: Optional[Dict[str, float]] = None,
        grid_config: Optional[Dict[str, Any]] = None,
        precomputation_manager = None,
    ):
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)
        self.derivative = derivative
        self.n_hedging_instruments = n_hedging_instruments
        self.precomputation_manager = precomputation_manager  # ← STORE IT HERE
    
        # Detect floating-grid mode
        self.is_floating_grid = bool(grid_config and grid_config.get("instruments", {}).get("floating_grid", {}).get("enabled", False))
    
        if self.is_floating_grid:
            from src.option_greek.vanilla import VanillaOption
            
            # Extract precomputation manager
            if precomputation_manager is None:
                # Fallback: try to get it from hedged derivative
                precomp_manager = getattr(derivative, 'precomp_manager', None)
                if precomp_manager is None:
                    raise ValueError("Floating grid requires precomputation_manager")
            else:
                precomp_manager = precomputation_manager
            
            self.grid_manager = FloatingGridManager(
                config=grid_config,
                derivative_class=VanillaOption,
                sim_params={"r": sim.r / 252.0, "T": sim.T, "N": sim.N, "M": sim.M},
                precomputation_manager=precomp_manager,
                garch_params=garch_params
            )
            self.position_ledger: Optional[VectorizedPositionLedger] = None
            self.hedging_derivatives_static = None
            # For logging/transaction typing
            self.instrument_types_list = ["stock"] + ["vanilla_option"] * self.grid_manager.grid_size
            self.obs_dim = 5  # Keep base obs_dim at 5 (time, S, S_prev, vol, V)
            # Note: action recurrence handled in network input, not obs_dim
            self.sparsity_penalty = grid_config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
            self.min_trade_size = grid_config["instruments"]["floating_grid"].get("min_trade_size", 0.1)
            logger.info(f"Floating grid mode: {self.grid_manager.grid_size} option buckets, obs_dim={self.obs_dim}")
        else:
            # Static mode
            self.grid_manager = None
            self.position_ledger = None
            self.hedging_derivatives_static = hedging_derivatives
            self.instrument_types_list = self._classify_instruments()
            self.obs_dim = 5
            self.sparsity_penalty = 0.0
            self.min_trade_size = 0.0
            logger.info(f"Static grid mode: {n_hedging_instruments} instruments, obs_dim={self.obs_dim}")
    
        # GARCH parameters (defaults provided if not present)
        self.garch_params = garch_params or {
            "omega": 1.593749e-07,
            "alpha": 2.308475e-06,
            "beta": 0.689984,
            "gamma": 342.870019,
            "lambda": 0.420499,
            "sigma0": sim.sigma,
        }
        self.omega = float(self.garch_params["omega"])
        self.alpha = float(self.garch_params["alpha"])
        self.beta = float(self.garch_params["beta"])
        self.gamma = float(self.garch_params["gamma"])
        self.lambda_ = float(self.garch_params["lambda"])
    
        # Market / contract parameters
        self.S0 = self.sim.S0
        self.K = torch.tensor(self.sim.K, dtype=torch.float32, device=self.device)
        self.T = self.sim.T
        self.r = self.sim.r / 252.0
        self.option_type = getattr(self.sim, "option_type", "call")
        self.side = self.sim.side
        self.contract_size = self.sim.contract_size
    
        # Transaction costs per instrument type
        default_tcp = getattr(self.sim, "TCP", 0.0001)
        self.transaction_costs = transaction_costs or {
            "stock": default_tcp,
            "vanilla_option": default_tcp * 10,
            "barrier_option": default_tcp * 20,
            "american_option": default_tcp * 15,
        }
    
        # Static-mode instrument metadata
        if not self.is_floating_grid and self.hedging_derivatives_static:
            self.instrument_maturities = []
            self.instrument_strikes = []
            self.instrument_types = []
            # lazy imports for type-checking
            from src.option_greek.vanilla import VanillaOption
            from src.option_greek.barrier import BarrierOption
            from src.option_greek.american import AmericanOption
    
            for deriv in self.hedging_derivatives_static:
                if deriv is None:
                    self.instrument_maturities.append(0)
                    self.instrument_strikes.append(0)
                    self.instrument_types.append("stock")
                else:
                    self.instrument_maturities.append(getattr(deriv, "N", self.N))
                    self.instrument_strikes.append(getattr(deriv, "K", self.K))
                    if isinstance(deriv, VanillaOption):
                        self.instrument_types.append("vanilla")
                    elif isinstance(deriv, BarrierOption):
                        self.instrument_types.append("barrier")
                    elif isinstance(deriv, AmericanOption):
                        self.instrument_types.append("american")
                    else:
                        self.instrument_types.append("vanilla")
    
        # Initial volatility / variance state
        self.sigma0 = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), self.sigma0, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2) / 252.0

    def _classify_instruments(self) -> List[str]:
        """Classify each hedging instrument (static mode only)."""
        # Lazy imports to avoid circular import problems.
        from src.option_greek.vanilla import VanillaOption  # type: ignore
        from src.option_greek.barrier import BarrierOption  # type: ignore
        from src.option_greek.american import AmericanOption  # type: ignore

        instrument_types = []
        for deriv in self.hedging_derivatives_static:
            if deriv is None:
                instrument_types.append("stock")
            elif isinstance(deriv, AmericanOption):
                instrument_types.append("american_option")
            elif isinstance(deriv, BarrierOption):
                instrument_types.append("barrier_option")
            else:
                instrument_types.append("vanilla_option")
        return instrument_types

    def _get_transaction_cost_rate(self, instrument_idx: int) -> float:
        instrument_type = self.instrument_types_list[instrument_idx]
        return self.transaction_costs.get(instrument_type, 0.0001)

    def reset(self):
        """
        Reset random shocks, variance and ledger state before an episode.
        """
        self.Z = torch.randn((self.M, self.N), dtype=torch.float32, device=self.device)
        sigma0_annual = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), sigma0_annual, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2) / 252.0

        # Reset ledger for floating grid mode
        if self.is_floating_grid:
            self.position_ledger = VectorizedPositionLedger(self.M, self.device)

        # Reset barrier statuses if those wrappers exist
        try:
            from src.option_greek.barrier_wrapper import BarrierOptionWithVanillaFallback  # type: ignore

            if isinstance(self.derivative, BarrierOptionWithVanillaFallback):
                self.derivative.reset_barrier_status()
            if not self.is_floating_grid and self.hedging_derivatives_static:
                for deriv in self.hedging_derivatives_static:
                    if isinstance(deriv, BarrierOptionWithVanillaFallback):
                        deriv.reset_barrier_status()
        except Exception:
            # If the wrapper isn't available, skip reset.
            pass

    # -------------------------
    # Trajectory simulation for policy rollout (returns observations + RL actions)
    # -------------------------
    def simulate_trajectory_and_get_observations(self, policy_net: PolicyNetGARCH):
        if self.is_floating_grid:
            return self._simulate_floating_grid(policy_net)
        else:
            return self._simulate_static_grid(policy_net)

    def _simulate_static_grid(self, policy_net: PolicyNetGARCH):
        """
        Static grid simulation: the policy outputs target positions for each instrument
        at each timestep. Returns:
            S_trajectory: [M, N+1]
            V_trajectory: [M, N+1]
            O_trajectories: list of [M, N+1] for each option hedging instrument (excluding stock)
            obs_sequence: [M, N+1, obs_dim]
            all_positions: [M, N+1, n_instruments]
        """
        S_trajectory = []
        V_trajectory = []
        O_trajectories = [[] for _ in range(len(self.hedging_derivatives_static) - 1)]
        obs_list = []

        # Initial states
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        S_trajectory.append(S_t)
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)

        # Price initial hedging instruments
        for i, hedge_deriv in enumerate(self.hedging_derivatives_static[1:]):
            K_hedge = getattr(hedge_deriv, "K", self.K)
            N_hedge = getattr(hedge_deriv, "N", self.N)
            O0 = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=0, N=N_hedge, h0=h0_current)
            O_trajectories[i].append(O0)

        # Initial observation
        obs_t = torch.zeros((self.M, 1, self.obs_dim), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0  # time
        obs_t[:, 0, 1] = S_t / self.K
        obs_t[:, 0, 2] = 0.5  # placeholder for previous normalized spot or other signal
        obs_t[:, 0, 3] = V0 / S_t
        obs_t[:, 0, 4] = self.side * V0
        obs_list.append(obs_t)

        # Get initial positions from the policy
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))
        outputs = []
        for head in policy_net.instrument_heads:
            output = head(x).squeeze(-1)[:, 0]  # [M]
            outputs.append(output)
        positions_t = torch.stack(outputs, dim=-1)  # [M, n_instruments]
        all_positions = [positions_t]

        # Time stepping
        for t in range(self.N):
            # Evolve GARCH variance and stock
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            # Price hedged derivative and hedging instruments
            h0_current = h_t.mean().item()
            V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t + 1, N=self.N, h0=h0_current)
            for i, hedge_deriv in enumerate(self.hedging_derivatives_static[1:]):
                K_hedge = getattr(hedge_deriv, "K", self.K)
                N_hedge = getattr(hedge_deriv, "N", self.N)
                O_t = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=t + 1, N=N_hedge, h0=h0_current)
                O_trajectories[i].append(O_t)

            S_trajectory.append(S_t)
            V_trajectory.append(V_t)

            # Observation for next timestep
            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, self.obs_dim), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t / self.K
            obs_new[:, 0, 2] = positions_t[:, 0].detach()  # use stock position as previous instrument signal
            obs_new[:, 0, 3] = V_t / S_t
            obs_new[:, 0, 4] = self.side * V_t
            obs_list.append(obs_new)

            # Run policy forward (conditioning on previous hidden_state)
            lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
            x = lstm_out
            for fc in policy_net.fc_layers:
                x = F.relu(fc(x))
            outputs = []
            for head in policy_net.instrument_heads:
                output = head(x).squeeze(-1)[:, 0]
                outputs.append(output)
            positions_t = torch.stack(outputs, dim=-1)
            all_positions.append(positions_t)

        # Collate outputs
        S_trajectory = torch.stack(S_trajectory, dim=1)  # [M, N+1]
        V_trajectory = torch.stack(V_trajectory, dim=1)  # [M, N+1]
        O_trajectories = [torch.stack(traj, dim=1) for traj in O_trajectories]  # list of [M, N+1]
        all_positions = torch.stack(all_positions, dim=1)  # [M, N+1, n_instruments]
        obs_sequence = torch.cat(obs_list, dim=1)  # [M, N+1, obs_dim]
        return S_trajectory, V_trajectory, O_trajectories, obs_sequence, all_positions
    
    def _simulate_floating_grid(self, policy_net: PolicyNetGARCH):
        """
        Floating grid simulation with action recurrence (Mueller et al. approach).
        The policy outputs:
          - actions[:, t, 0] = target stock position
          - actions[:, t, i] (i>=1) = trades for option bucket i-1 (rounded to integer)
        Returns:
            S_trajectory: [M, N+1]
            V_trajectory: [M, N+1]
            None (in place of O_trajectories)
            obs_sequence: [M, N+1, obs_dim]
            all_actions: [M, N+1, n_instruments]
        """
        S_trajectory = []
        V_trajectory = []
        obs_list = []
        all_actions = []
    
        # Initialize ledger and state
        ledger = VectorizedPositionLedger(self.M, self.device)
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        stock_position = torch.zeros(self.M, dtype=torch.float32, device=self.device)
    
        S_trajectory.append(S_t.clone())
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)
    
        # Initial observation
        obs_t = torch.zeros((self.M, 1, self.obs_dim), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0  # time (normalized)
        obs_t[:, 0, 1] = S_t  # current spot
        obs_t[:, 0, 2] = self.S0  # previous spot (t=0 use S0)
        obs_t[:, 0, 3] = torch.sqrt(h_t)  # volatility
        obs_t[:, 0, 4] = V0  # derivative value
        obs_list.append(obs_t)
    
        # Initialize previous actions as zeros for t=0
        prev_actions_t = torch.zeros(
            (self.M, 1, self.n_hedging_instruments), 
            dtype=torch.float32, 
            device=self.device
        )
    
        # Get initial actions WITH action recurrence
        # NOTE: Pass hidden_states as positional argument, NOT keyword
        # Use torch.no_grad() for trajectory simulation to prevent memory buildup
        with torch.no_grad():
            if policy_net.use_action_recurrence:
                outputs, hidden_states = policy_net(obs_t, prev_actions_t, None)  # None = initial hidden states
            else:
                outputs, hidden_states = policy_net(obs_t, None, None)  # Ignore prev_actions if not using recurrence
            
            actions_t = torch.stack([out[:, 0] for out in outputs], dim=-1)  # [M, n_instruments]
        all_actions.append(actions_t)
    
        # Main loop
        for t in range(self.N):
            # Execute actions from previous output (actions_t is target positions / trades)
            actions_prev = all_actions[-1]
    
            # Stock semantics: interpret as target position
            stock_trade = actions_prev[:, 0] - stock_position
            stock_position = actions_prev[:, 0]
    
            # Options: treat actions_prev[:, i] (i>=1) as immediate trade quantities (rounded)
            for bucket_idx in range(1, self.n_hedging_instruments):
                trade_qty = torch.round(actions_prev[:, bucket_idx])  # integerize
                # suppress small trades
                trade_qty = torch.where(
                    trade_qty.abs() < self.min_trade_size, 
                    torch.zeros_like(trade_qty), 
                    trade_qty
                )
                
                if trade_qty.abs().sum() > 1e-6:
                    # Get bucket parameters (moneyness and maturity)
                    moneyness, maturity_days = self.grid_manager.get_bucket_params(bucket_idx - 1)
                    
                    # VECTORIZED: Compute strikes for ALL paths at once
                    K_paths = moneyness * S_t  # [M]
                    
                    # Create derivative template
                    from src.option_greek.vanilla import VanillaOption
                    deriv_template = VanillaOption(
                        precomputation_manager=self.precomputation_manager,
                        garch_params=self.garch_params,
                        option_type=self.grid_manager.option_type
                    )
                    deriv_template.N = maturity_days
                    deriv_template.created_at_step = t
                    
                    # Add positions to ledger (BATCHED - NO LOOP)
                    nonzero_mask = trade_qty.abs() > 1e-6
                    
                    if nonzero_mask.sum() > 0:
                        ledger.add_positions_batch(
                            path_indices=torch.where(nonzero_mask)[0],
                            quantities=trade_qty[nonzero_mask],
                            strikes=K_paths[nonzero_mask],
                            expiry_steps=torch.full(
                                (nonzero_mask.sum(),), 
                                t + maturity_days, 
                                dtype=torch.long, 
                                device=self.device
                            ),
                            created_at_step=t,
                            original_maturity=maturity_days,
                            bucket_idx=bucket_idx,
                            opening_spots=S_t[nonzero_mask],
                            derivative_template=deriv_template
                        )
    
            # Evolve GARCH and stock
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
    
            # Remove expired positions (they expire at t+1 if expiry_step <= t+1)
            ledger.remove_expired(t + 1)
    
            # Price current hedged derivative
            h0_current = h_t.mean().item()
            V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t + 1, N=self.N, h0=h0_current)
            S_trajectory.append(S_t.clone())
            V_trajectory.append(V_t)
    
            # Build observation for next iteration
            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, self.obs_dim), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t
            # previous spot is the one before the last append
            obs_new[:, 0, 2] = S_trajectory[-2]
            obs_new[:, 0, 3] = torch.sqrt(h_t)
            obs_new[:, 0, 4] = V_t
            obs_list.append(obs_new)
    
            # Ask policy for next actions
            # Prepare previous actions for network input (Mueller's approach)
            if policy_net.use_action_recurrence:
                # CRITICAL: Use .detach() on prev_actions to prevent BPTT through time
                prev_actions_t = actions_t.detach().unsqueeze(1)
                # Detach hidden states - hidden_states is a list of (h, c) tuples
                hidden_states = [(h.detach(), c.detach()) for h, c in hidden_states]
                # Pass as positional arguments: obs, prev_actions, hidden_states
                outputs, hidden_states = policy_net(obs_new, prev_actions_t, hidden_states)
            else:
                # Still detach hidden state even without action recurrence
                hidden_states = [(h.detach(), c.detach()) for h, c in hidden_states]
                # Pass as positional arguments: obs, prev_actions, hidden_states
                outputs, hidden_states = policy_net(obs_new, None, hidden_states)
            
            actions_t = torch.stack([out[:, 0] for out in outputs], dim=-1)
            all_actions.append(actions_t)
        
        # Stack outputs and store ledger
        S_trajectory = torch.stack(S_trajectory, dim=1)  # [M, N+1]
        V_trajectory = torch.stack(V_trajectory, dim=1)  # [M, N+1]
        all_actions = torch.stack(all_actions, dim=1)  # [M, N+1, n_instruments]
        obs_sequence = torch.cat(obs_list, dim=1)  # [M, N+1, obs_dim]
    
        self.position_ledger = ledger
        return S_trajectory, V_trajectory, None, obs_sequence, all_actions
    # -------------------------
    # Full-trajectory P&L simulation using policy outputs
    # -------------------------
    def simulate_full_trajectory(self, all_actions: torch.Tensor, O_trajectories_or_None):
        if self.is_floating_grid:
            return self._simulate_full_trajectory_floating(all_actions)
        else:
            return self._simulate_full_trajectory_static(all_actions, O_trajectories_or_None)


    def _simulate_full_trajectory_floating(self, all_actions: torch.Tensor):
        """
        P&L calculation for floating-grid mode with PATH-SPECIFIC strikes.
        FULLY VECTORIZED VERSION.
        """
        ledger = VectorizedPositionLedger(self.M, self.device)
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        stock_position = torch.zeros(self.M, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        h0_current = h_t.mean().item()
        
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        B_t = self.side * V0
        V0_portfolio = self.side * V0
        
        # Preallocate trajectories (optimization)
        S_trajectory = torch.zeros(self.M, self.N + 1, dtype=torch.float32, device=self.device)
        B_trajectory = torch.zeros(self.M, self.N + 1, dtype=torch.float32, device=self.device)
        stock_pos_trajectory = torch.zeros(self.M, self.N + 1, dtype=torch.float32, device=self.device)
        
        S_trajectory[:, 0] = S_t
        B_trajectory[:, 0] = B_t
        stock_pos_trajectory[:, 0] = stock_position
        
        cost_breakdown = {
            "stock": torch.zeros(self.M, device=self.device),
            "vanilla_option": torch.zeros(self.M, device=self.device)
        }
        soft_constraint_violations = torch.zeros(self.M, device=self.device)
        ledger_size_trajectory = []
        
        compute_soft_constraint = (
            hasattr(self, 'lambda_constraint') and 
            self.lambda_constraint is not None and 
            self.lambda_constraint > 0
        )
        
        tcp_stock = self._get_transaction_cost_rate(0)
        
        for t in range(self.N):
            actions_t = all_actions[:, t]
            
            # ===== STOCK TRADE (vectorized) =====
            stock_trade = actions_t[:, 0] - stock_position
            stock_cost = tcp_stock * torch.abs(stock_trade) * S_t
            B_t -= stock_trade * S_t + stock_cost
            cost_breakdown["stock"] += stock_cost
            stock_position = actions_t[:, 0]
            
            # ===== OPTION TRADES (VECTORIZED ACROSS BUCKETS) =====
            for bucket_idx in range(1, self.n_hedging_instruments):
                trade_qty = torch.round(actions_t[:, bucket_idx])
                trade_qty = torch.where(
                    trade_qty.abs() < self.min_trade_size,
                    torch.zeros_like(trade_qty),
                    trade_qty
                )
                
                if trade_qty.abs().sum() < 1e-6:
                    continue
                
                moneyness, maturity_days = self.grid_manager.get_bucket_params(bucket_idx - 1)
                tcp_option = self._get_transaction_cost_rate(bucket_idx)
                
                # VECTORIZED: Compute strikes for ALL paths at once
                K_paths = moneyness * S_t  # [M]
                
                # Create derivative template
                from src.option_greek.vanilla import VanillaOption
                deriv_template = VanillaOption(
                    precomputation_manager=self.precomputation_manager,
                    garch_params=self.garch_params,
                    option_type=self.grid_manager.option_type
                )
                deriv_template.N = maturity_days
                deriv_template.created_at_step = t
                
                # VECTORIZED: Price ALL paths at once
                option_prices = deriv_template.price(
                    S=S_t,
                    K=K_paths,
                    step_idx=0,
                    N=maturity_days,
                    h0=h0_current
                )
                
                # VECTORIZED: Compute costs and update bank
                option_costs = tcp_option * trade_qty.abs() * option_prices
                B_t -= trade_qty * option_prices + option_costs
                cost_breakdown["vanilla_option"] += option_costs
                
                # Add positions to ledger (BATCHED - NO LOOP)
                nonzero_mask = trade_qty.abs() > 1e-6
                
                if nonzero_mask.sum() > 0:
                    ledger.add_positions_batch(
                        path_indices=torch.where(nonzero_mask)[0],
                        quantities=trade_qty[nonzero_mask],
                        strikes=K_paths[nonzero_mask],
                        expiry_steps=torch.full(
                            (nonzero_mask.sum(),), 
                            t + maturity_days, 
                            dtype=torch.long, 
                            device=self.device
                        ),
                        created_at_step=t,
                        original_maturity=maturity_days,
                        bucket_idx=bucket_idx,
                        opening_spots=S_t[nonzero_mask],
                        derivative_template=deriv_template
                    )
            
            # ===== EVOLVE GARCH AND STOCK =====
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            # ===== ACCRUE INTEREST =====
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)
            
            # ===== EXPIRATIONS: Realize payoffs (VECTORIZED) =====
            h0_current = h_t.mean().item()
            expired_payoffs = ledger.compute_and_remove_expired(S_t, t + 1, h0_current)
            B_t += expired_payoffs
            
            # ===== SOFT CONSTRAINT (VECTORIZED) =====
            if compute_soft_constraint:
                portfolio_value = ledger.compute_total_value(S_t, t + 1, h0_current)
                P_t = B_t + stock_position * S_t + portfolio_value
                
                V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t + 1, N=self.N, h0=h0_current)
                V_t_phi = self.side * V_t_phi
                xi_t = P_t - V_t_phi
                soft_constraint_violations += torch.clamp(xi_t, min=0.0)
            
            # ===== RECORD TRAJECTORIES =====
            S_trajectory[:, t + 1] = S_t
            B_trajectory[:, t + 1] = B_t
            stock_pos_trajectory[:, t + 1] = stock_position
            ledger_size_trajectory.append(float(ledger.get_ledger_size().mean().item()))
        
        # ===== TERMINAL PAYOFF =====
        if hasattr(self.derivative, "option_type"):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            payoff = torch.clamp(S_t - self.K, min=0.0)
        payoff = payoff * self.contract_size
        
        # ===== TERMINAL PORTFOLIO VALUE (VECTORIZED) =====
        portfolio_value_final = ledger.compute_total_value(S_t, self.N, h_t.mean().item())
        terminal_value = B_t + stock_position * S_t + portfolio_value_final
        terminal_error = terminal_value - self.side * payoff
        
        trajectories = {
            "S": S_trajectory,
            "B": B_trajectory,
            "stock_positions": stock_pos_trajectory,
            "positions": all_actions,
            "O": None,
            "cost_breakdown": cost_breakdown,
            "soft_constraint_violations": soft_constraint_violations,
            "V0_portfolio": V0_portfolio,
            "ledger_size_trajectory": ledger_size_trajectory,
            "all_actions": all_actions,
        }
        
        return terminal_error, trajectories
    # Loss & training utilities
# ----------------------------------------
def compute_loss_with_soft_constraint(
    terminal_errors: torch.Tensor,
    trajectories: Dict[str, Any],
    risk_measure: str = "mse",
    alpha: Optional[float] = None,
    lambda_constraint: float = 0.0,
    lambda_sparsity: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined loss with risk measure, soft constraint, and sparsity penalty.
    
    Returns:
        total_loss, risk_loss, constraint_penalty, sparsity_penalty
    """
    M = terminal_errors.shape[0]
    
    # Compute risk loss based on chosen measure
    if risk_measure == "mse":
        risk_loss = (terminal_errors ** 2).mean()
    elif risk_measure == "smse":
        positive_mask = (terminal_errors >= 0).float()
        risk_loss = ((terminal_errors ** 2) * positive_mask).mean()
    elif risk_measure == "cvar":
        if alpha is None:
            raise ValueError("alpha parameter required for CVaR")
        sorted_errors, _ = torch.sort(terminal_errors, descending=True)
        n_tail = max(1, int(np.ceil(M * (1 - alpha))))
        risk_loss = sorted_errors[:n_tail].mean()
    elif risk_measure == "var":
        if alpha is None:
            raise ValueError("alpha parameter required for VaR")
        risk_loss = torch.quantile(terminal_errors, alpha)
    elif risk_measure == "mae":
        risk_loss = terminal_errors.abs().mean()
    else:
        raise ValueError(f"Unknown risk measure: {risk_measure}")
    
    # Compute soft constraint penalty
    soft_violations = trajectories.get("soft_constraint_violations", torch.zeros_like(terminal_errors))
    constraint_penalty = soft_violations.mean()
    
    # Compute sparsity penalty (for floating grid mode) - SCALED VERSION
    sparsity_penalty = torch.tensor(0.0, device=terminal_errors.device)
    if lambda_sparsity > 0 and "all_actions" in trajectories:
        all_actions = trajectories["all_actions"]  # [M, N+1, n_instruments]
        option_actions = all_actions[:, :, 1:]  # exclude stock (index 0)
        
        # Raw sparsity: mean absolute number of contracts
        raw_sparsity = option_actions.abs().mean()
        
        # Scale to risk magnitude (makes lambda interpretable)
        # lambda=0.01 now means "1% penalty per contract relative to risk"
        # Use detach() so sparsity gradient doesn't affect risk_loss
        sparsity_penalty = raw_sparsity * risk_loss.detach()
    
    # Total loss with properly scaled sparsity
    total_loss = risk_loss + lambda_constraint * constraint_penalty + lambda_sparsity * sparsity_penalty
    
    return total_loss, risk_loss, constraint_penalty, sparsity_penalty
def get_transaction_costs(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract transaction costs from config."""
    tc = config.get("transaction_costs", {})
    return {
        "stock": tc.get("stock", 0.0001),
        "vanilla_option": tc.get("vanilla_option", 0.001),
        "barrier_option": tc.get("barrier_option", 0.002),
        "american_option": tc.get("american_option", 0.0015),
    }
# def train_episode(
#     episode: int,
#     config: Dict[str, Any],
#     policy_net: PolicyNetGARCH,
#     optimizer: torch.optim.Optimizer,
#     hedged_derivative,
#     hedging_derivatives,
#     HedgingSim,
#     device: torch.device,
#     precomputation_manager,
# ) -> Dict[str, Any]:
#     """Train for a single episode with configurable risk measure and soft constraint."""
    
#     hedged_cfg = config["hedged_option"]
#     mode = config["instruments"].get("mode", "static")
#     is_floating_grid = (mode == "floating_grid")
    
#     # Get transaction costs from config
#     transaction_costs = get_transaction_costs(config)
    
#     # NEW: Get risk measure and soft constraint configuration
#     risk_config = config.get("risk_measure", {"type": "mse"})
#     constraint_config = config.get("soft_constraint", {"enabled": False, "lambda": 0.0})
    
#     risk_measure = risk_config.get("type", "mse")
#     alpha = risk_config.get("alpha", None)
#     lambda_constraint = constraint_config.get("lambda", 0.0) if constraint_config.get("enabled", False) else 0.0
    
#     # Get sparsity penalty for floating grid
#     lambda_sparsity = 0.0
#     if is_floating_grid:
#         lambda_sparsity = config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
    
#     # FIX: Convert K to tensor on device BEFORE passing to HedgingSim
#     K_tensor = torch.tensor(hedged_cfg["K"], dtype=torch.float32, device=device)
#     S0_tensor = torch.tensor(config["simulation"]["S0"], dtype=torch.float32, device=device)
    
#     sim = HedgingSim(
#         S0=S0_tensor,  # Pass as tensor
#         K=K_tensor,    # Pass as tensor
#         m=0.1,
#         r=config["simulation"]["r"],
#         sigma=config["garch"]["sigma0"],
#         T=config["simulation"]["T"],
#         option_type=hedged_cfg["option_type"],
#         position=hedged_cfg["side"],
#         M=config["simulation"]["M"],
#         N=config["simulation"]["N"],
#         TCP=transaction_costs.get('stock', 0.0001),
#         seed=episode
#     )

#     env = HedgingEnvGARCH(
#         sim=sim,
#         derivative=hedged_derivative,
#         hedging_derivatives=None if is_floating_grid else hedging_derivatives,
#         garch_params=config["garch"],
#         n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
#         dt_min=config["environment"]["dt_min"],
#         device=str(device),
#         transaction_costs=transaction_costs,
#         grid_config=config if is_floating_grid else None,
#         precomputation_manager=precomputation_manager,
#         lambda_constraint=lambda_constraint  # ← Add this
#     )

#     env.reset()
#     S_traj, V_traj, O_traj, obs_sequence, RL_actions = env.simulate_trajectory_and_get_observations(policy_net)
#     terminal_errors, trajectories = env.simulate_full_trajectory(RL_actions, O_traj)

#     optimizer.zero_grad()
#     total_loss, risk_loss, constraint_penalty, sparsity_pen = compute_loss_with_soft_constraint(
#         terminal_errors, trajectories, risk_measure=risk_measure, alpha=alpha, lambda_constraint=lambda_constraint, lambda_sparsity=lambda_sparsity
#     )
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=config["training"]["gradient_clip_max_norm"])
#     optimizer.step()

#     if torch.isnan(total_loss) or torch.isinf(total_loss):
#         logging.error("Loss became NaN/Inf")
#         raise RuntimeError("Loss became NaN/Inf")

#     final_reward = -float(total_loss.item())

#     # Log summary
#     log_msg = (
#         f"Episode {episode} | Reward: {final_reward:.6f} | "
#         f"Total Loss: {total_loss.item():.6f} | Risk Loss: {risk_loss.item():.6f}"
#     )
#     if lambda_constraint > 0:
#         avg_violation = trajectories["soft_constraint_violations"].mean().item()
#         log_msg += f" | Constraint: {constraint_penalty.item():.6f} (Avg Viol: {avg_violation:.6f})"
#     if lambda_sparsity > 0:
#         log_msg += f" | Sparsity: {sparsity_pen.item():.6f}"
#     if is_floating_grid and "ledger_size_trajectory" in trajectories:
#         avg_ledger = np.mean(trajectories["ledger_size_trajectory"])
#         max_ledger = np.max(trajectories["ledger_size_trajectory"])
#         log_msg += f" | Ledger (Avg/Max): {avg_ledger:.1f}/{max_ledger:.0f}"
#     logging.info(log_msg)

#     return {
#         "episode": episode,
#         "loss": total_loss.item(),
#         "risk_loss": risk_loss.item(),
#         "constraint_penalty": constraint_penalty.item(),
#         "sparsity_penalty": sparsity_pen.item() if isinstance(sparsity_pen, torch.Tensor) else float(sparsity_pen),
#         "reward": final_reward,
#         "trajectories": trajectories,
#         "RL_positions": RL_actions,
#         "S_traj": S_traj,
#         "V_traj": V_traj,
#         "O_traj": O_traj,
#         "env": env,
#     }
