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
# Policy network (LSTM + per-instrument heads)
# ----------------------------------------
class PolicyNetGARCH(nn.Module):
    """
    Policy network that consumes a sequence of observations (batch_first=True)
    and outputs a scalar per hedging instrument at each timestep.

    Architecture:
      - LSTM (obs_dim -> hidden_size), num_layers configurable
      - N fully-connected layers (applied to LSTM outputs)
      - One linear head per hedging instrument (outputs scalar per time step)
    """

    def __init__(
        self,
        obs_dim: int = 5,
        hidden_size: int = 128,
        n_hedging_instruments: int = 2,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
    ):
        super().__init__()
        self.n_hedging_instruments = n_hedging_instruments
        self.hidden_size = hidden_size

        # LSTM
        self.lstm = nn.LSTM(
            input_size=obs_dim, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True
        )

        # A small stack of fully connected layers applied to the LSTM outputs
        self.fc_layers = nn.ModuleList()
        in_dim = hidden_size
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size

        # One head per hedging instrument
        self.instrument_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_hedging_instruments)])

        # Initialize parameters
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, obs_sequence: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            obs_sequence: [batch, seq_len, obs_dim]
        Returns:
            List of length n_hedging_instruments, each tensor is [batch, seq_len]
            (i.e., the head outputs per time-step). This matches the original
            code which returned per-head sequences.
        """
        lstm_out, _ = self.lstm(obs_sequence)  # [batch, seq_len, hidden_size]
        x = lstm_out
        for fc in self.fc_layers:
            # apply fc to last dim
            x = F.relu(fc(x))
        # produce per-head outputs, squeeze last dim
        outputs = [head(x).squeeze(-1) for head in self.instrument_heads]  # list of [batch, seq_len]
        return outputs


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
        logger.info(
            f"FloatingGridManager initialized: {len(self.moneyness_levels)} moneyness levels × "
            f"{len(self.maturity_days)} maturities = {self.grid_size} option buckets"
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
class PositionLedger:
    """
    Tracks per-path option positions (for the floating grid mode).
    Each path has an independent list of open positions (with quantity, strike, expiry).
    """

    def __init__(self, M: int, device: torch.device):
        self.M = M
        self.device = device
        self.positions: List[List[Dict[str, Any]]] = [[] for _ in range(M)]

    def add_position(self, derivative, quantity: torch.Tensor, bucket_idx: int, current_step: int, S_current: torch.Tensor):
        """
        Add positions per-path.

        Args:
            derivative: option object (shared config; pricing methods operate pathwise)
            quantity: [M] tensor (can be negative for short)
            bucket_idx: integer bucket index (with stock reserved at index 0)
            current_step: timestep when opened
            S_current: [M] spots at opening
        """
        for path_idx in range(self.M):
            qty = float(quantity[path_idx])
            if abs(qty) > 1e-6:
                self.positions[path_idx].append({
                    "derivative": derivative,
                    "quantity": qty,
                    "bucket_idx": bucket_idx,
                    "created_at_step": current_step,  # ← ADD THIS
                    "expiry_step": current_step + derivative.N,
                    "strike": derivative.K,
                    "original_maturity_N": derivative.N,  # ← ADD THIS
                    "opening_spot": float(S_current[path_idx]),
                })

    def remove_expired(self, current_step: int):
        """Remove positions whose expiry_step <= current_step (they've expired)."""
        for path_idx in range(self.M):
            self.positions[path_idx] = [pos for pos in self.positions[path_idx] if pos["expiry_step"] > current_step]

    def compute_and_remove_expired(self, S_current: torch.Tensor, current_step: int, h0: float) -> torch.Tensor:
        """
        Compute payoffs for positions that expire at or before current_step,
        remove them from ledger, and return the per-path total payoffs.
        """
        payoffs = torch.zeros(self.M, device=self.device)
        for path_idx in range(self.M):
            expired = []
            remaining = []
            for pos in self.positions[path_idx]:
                if pos["expiry_step"] <= current_step:
                    expired.append(pos)
                else:
                    remaining.append(pos)
            for pos in expired:
                S_path = float(S_current[path_idx].item())
                K = pos["strike"]
                opt_type = getattr(pos["derivative"], "option_type", "call").lower()
                if opt_type == "call":
                    payoff = max(S_path - K, 0.0)
                else:
                    payoff = max(K - S_path, 0.0)
                payoffs[path_idx] += pos["quantity"] * payoff
            self.positions[path_idx] = remaining
        return payoffs

    def compute_total_value(self, S_current: torch.Tensor, current_step: int, h0: float) -> torch.Tensor:
        """
        Price all active positions in the ledger (for each path) and return aggregate [M].
        This uses each stored pos['derivative'] to compute current prices (per-path).
        """
        portfolio_value = torch.zeros(self.M, device=self.device)
        for path_idx in range(self.M):
            for pos in self.positions[path_idx]:
                if pos["expiry_step"] > current_step:
                    S_path = S_current[path_idx].unsqueeze(0)  # shape [1]
                    step_idx = current_step - pos["created_at_step"]  # ← ADD THIS
                    price = pos["derivative"].price(
                        S=S_path, 
                        K=pos["strike"], 
                        step_idx=step_idx,  # ← CHANGE FROM current_step
                        N=pos["original_maturity_N"],  # ← CHANGE FROM derivative.N
                        h0=h0
                    )
                    portfolio_value[path_idx] += pos["quantity"] * float(price)
        return portfolio_value

    def compute_greeks(self, S_current: torch.Tensor, current_step: int, h0: float, greek_name: str) -> torch.Tensor:
        """
        Aggregate a named Greek across all active positions per path.
        greek_name must be one of 'delta', 'gamma', 'vega', 'theta' and the derivative must expose it.
        """
        greeks = torch.zeros(self.M, device=self.device)
        for path_idx in range(self.M):
            for pos in self.positions[path_idx]:
                if pos["expiry_step"] > current_step:
                    S_path = S_current[path_idx].unsqueeze(0)
                    greek_method = getattr(pos["derivative"], greek_name)
                    greek_val = greek_method(S=S_path, K=pos["strike"], step_idx=current_step, N=pos["derivative"].N, h0=h0)
                    greeks[path_idx] += float(greek_val)
        return greeks

    def get_ledger_size(self) -> torch.Tensor:
        sizes = torch.tensor([len(self.positions[path_idx]) for path_idx in range(self.M)], dtype=torch.float32, device=self.device)
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
    ):
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)
        self.derivative = derivative
        self.n_hedging_instruments = n_hedging_instruments

        # Detect floating-grid mode
        self.is_floating_grid = bool(grid_config and grid_config.get("instruments", {}).get("floating_grid", {}).get("enabled", False))

        if self.is_floating_grid:
            from src.option_greek.vanilla import VanillaOption
            
            # Get precomputation manager from the hedged derivative
            # (assumes hedged derivative is a VanillaOption with precomp_manager)
            precomp_manager = derivative.precomp_manager
            
            self.grid_manager = FloatingGridManager(
                config=grid_config,
                derivative_class=VanillaOption,
                sim_params={"r": sim.r / 252.0, "T": sim.T, "N": sim.N, "M": sim.M},
                precomputation_manager=precomp_manager,  # ADD THIS
                garch_params=garch_params  # ADD THIS
            )
            self.position_ledger: Optional[PositionLedger] = None  # created per episode in reset()
            self.hedging_derivatives_static = None
            # For logging/transaction typing
            self.instrument_types_list = ["stock"] + ["vanilla_option"] * self.grid_manager.grid_size
            self.obs_dim = 5
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
        self.r = self.sim.r / 252.0  # keep consistent with original code (per-step r)
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
            from src.option_greek.vanilla import VanillaOption  # type: ignore
            from src.option_greek.barrier import BarrierOption  # type: ignore
            from src.option_greek.american import AmericanOption  # type: ignore

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
            self.position_ledger = PositionLedger(self.M, self.device)

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
        Floating grid simulation. The policy outputs:
          - actions[:, t, 0] = target stock position
          - actions[:, t, i] (i>=1) = trades (or target) for option bucket i-1 (rounded to integer)
        The ledger records opened option positions (with expiries); P&L computed later using ledger.
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
        ledger = PositionLedger(self.M, self.device)
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

        # Get initial actions
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))
        outputs = []
        for head in policy_net.instrument_heads:
            output = head(x).squeeze(-1)[:, 0]
            outputs.append(output)
        actions_t = torch.stack(outputs, dim=-1)  # [M, n_instruments]
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
                trade_qty = torch.where(trade_qty.abs() < self.min_trade_size, torch.zeros_like(trade_qty), trade_qty)
                if trade_qty.abs().sum() > 1e-6:
                    S_mean = float(S_t.mean().item())
                    deriv = self.grid_manager.create_derivative(
                        S_current=S_mean, bucket_idx=bucket_idx - 1, current_step=t, total_steps=self.N
                    )
                    ledger.add_position(deriv, trade_qty, bucket_idx, t, S_t)

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

            # Ask policy for next actions (unless at the last time step)
            if t < self.N - 1:
                lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
                x = lstm_out
                for fc in policy_net.fc_layers:
                    x = F.relu(fc(x))
                outputs = []
                for head in policy_net.instrument_heads:
                    output = head(x).squeeze(-1)[:, 0]
                    outputs.append(output)
                actions_t = torch.stack(outputs, dim=-1)
                all_actions.append(actions_t)

        # Stack outputs and store ledger
        S_trajectory = torch.stack(S_trajectory, dim=1)
        V_trajectory = torch.stack(V_trajectory, dim=1)
        all_actions = torch.stack(all_actions, dim=1)  # [M, N+1, n_instruments]
        obs_sequence = torch.cat(obs_list, dim=1)

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

    def _simulate_full_trajectory_static(self, all_positions: torch.Tensor, O_trajectories: List[torch.Tensor]):
        """
        P&L calculation for static hedging instruments.
        all_positions: [M, N+1, n_instruments] (positions per time step)
        O_trajectories: list of option price paths [M, N+1] for the option instruments (excluding stock)
        """
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        positions_t = all_positions[:, 0]
        h0_current = self.h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)

        # Initialize bank account with derivative proceeds and the cost of initial hedges
        B_t = self.side * V0 - positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            B_t -= positions_t[:, i + 1] * O_trajectories[i][:, 0]

        V0_portfolio = self.side * V0
        h_t = self.h_t.clone()

        S_traj, B_traj = [S_t], [B_t]
        cost_breakdown = {
            "stock": torch.zeros(self.M, device=self.device),
            "vanilla_option": torch.zeros(self.M, device=self.device),
            "barrier_option": torch.zeros(self.M, device=self.device),
            "american_option": torch.zeros(self.M, device=self.device),
        }
        soft_constraint_violations = torch.zeros(self.M, device=self.device)

        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            # Evolve stock
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            # New positions
            positions_new = all_positions[:, t + 1]

            # Accrue interest on bank account
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)

            # Rebalance per instrument and apply transaction costs
            for i in range(self.n_hedging_instruments):
                trade = positions_new[:, i] - positions_t[:, i]
                tcp_rate = self._get_transaction_cost_rate(i)
                if i == 0:
                    price = S_t
                    instrument_type = "stock"
                else:
                    price = O_trajectories[i - 1][:, t + 1]
                    instrument_type = self.instrument_types_list[i]
                cost = tcp_rate * torch.abs(trade) * price
                B_t = B_t - trade * price - cost
                cost_breakdown[instrument_type] += cost

            positions_t = positions_new

            # Compute portfolio value
            P_t = B_t + positions_t[:, 0] * S_t
            for i in range(len(O_trajectories)):
                P_t += positions_t[:, i + 1] * O_trajectories[i][:, t + 1]

            # Soft constraint: positive exceeding of portfolio over derivative value
            h0_current = h_t.mean().item()
            V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t + 1, N=self.N, h0=h0_current)
            V_t_phi = self.side * V_t_phi
            xi_t = P_t - V_t_phi
            soft_constraint_violations += torch.clamp(xi_t, min=0.0)

            S_traj.append(S_t)
            B_traj.append(B_t)

        # Terminal payoff and portfolio value
        if hasattr(self.derivative, "option_type"):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            payoff = torch.clamp(S_t - self.K, min=0.0)
        payoff = payoff * self.contract_size

        terminal_value = B_t + positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            terminal_value += positions_t[:, i + 1] * O_trajectories[i][:, -1]

        terminal_error = terminal_value - self.side * payoff

        trajectories = {
            "S": torch.stack(S_traj, dim=1),
            "B": torch.stack(B_traj, dim=1),
            "positions": all_positions,
            "O": O_trajectories,
            "cost_breakdown": cost_breakdown,
            "soft_constraint_violations": soft_constraint_violations,
            "V0_portfolio": V0_portfolio,
        }
        return terminal_error, trajectories

    def _simulate_full_trajectory_floating(self, all_actions: torch.Tensor):
        """
        P&L calculation for the floating-grid mode. all_actions interpreted as:
            - all_actions[:, t, 0] = target stock position at time t
            - all_actions[:, t, i] (i>=1) = trade quantities for bucket (i-1) at time t
        The ledger is reconstructed and used to price portfolio positions pathwise.
        """
        ledger = PositionLedger(self.M, self.device)
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        stock_position = torch.zeros(self.M, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)

        # Initialize bank account with proceeds from selling/hedging the derivative
        B_t = self.side * V0
        V0_portfolio = self.side * V0

        S_traj, B_traj = [S_t], [B_t]
        stock_pos_traj = [stock_position.clone()]

        cost_breakdown = {"stock": torch.zeros(self.M, device=self.device), "vanilla_option": torch.zeros(self.M, device=self.device)}
        soft_constraint_violations = torch.zeros(self.M, device=self.device)
        ledger_size_trajectory: List[float] = []

        for t in range(self.N):
            actions_t = all_actions[:, t]

            # Stock trade (target semantics)
            stock_trade = actions_t[:, 0] - stock_position
            tcp_stock = self._get_transaction_cost_rate(0)
            stock_cost = tcp_stock * torch.abs(stock_trade) * S_t
            B_t -= stock_trade * S_t + stock_cost
            cost_breakdown["stock"] += stock_cost
            stock_position = actions_t[:, 0]

            # Option trades (interpreted as integerized trade quantities)
            for bucket_idx in range(1, self.n_hedging_instruments):
                trade_qty = torch.round(actions_t[:, bucket_idx])
                trade_qty = torch.where(trade_qty.abs() < self.min_trade_size, torch.zeros_like(trade_qty), trade_qty)
                if trade_qty.abs().sum() > 1e-6:
                    S_mean = float(S_t.mean().item())
                    deriv = self.grid_manager.create_derivative(S_current=S_mean, bucket_idx=bucket_idx - 1, current_step=t, total_steps=self.N)
                    # price for transaction cost per-path
                    option_prices = torch.zeros(self.M, device=self.device)
                    for path_idx in range(self.M):
                        if abs(trade_qty[path_idx]) > 1e-6:
                            S_path = S_t[path_idx].unsqueeze(0)
                            price = deriv.price(S=S_path, K=deriv.K, step_idx=t, N=deriv.N, h0=h0_current)
                            option_prices[path_idx] = float(price)
                    tcp_option = self._get_transaction_cost_rate(bucket_idx)
                    option_cost = tcp_option * trade_qty.abs() * option_prices
                    B_t -= trade_qty * option_prices + option_cost
                    cost_breakdown["vanilla_option"] += option_cost
                    ledger.add_position(deriv, trade_qty, bucket_idx, t, S_t)

            # Evolve GARCH and stock
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            # Accrue interest
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)

            # Expirations: realize payoffs for expired positions and remove them
            expired_payoffs = ledger.compute_and_remove_expired(S_t, t + 1, h_t.mean().item())
            B_t += expired_payoffs

            # Portfolio value (booked options via ledger)
            h0_current = h_t.mean().item()
            portfolio_value = ledger.compute_total_value(S_t, t + 1, h0_current)
            P_t = B_t + stock_position * S_t + portfolio_value

            # Soft constraint
            V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t + 1, N=self.N, h0=h0_current)
            V_t_phi = self.side * V_t_phi
            xi_t = P_t - V_t_phi
            soft_constraint_violations += torch.clamp(xi_t, min=0.0)

            # Record trajectories
            S_traj.append(S_t)
            B_traj.append(B_t)
            stock_pos_traj.append(stock_position.clone())
            ledger_size_trajectory.append(float(ledger.get_ledger_size().mean().item()))

        # Terminal payoff
        if hasattr(self.derivative, "option_type"):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            payoff = torch.clamp(S_t - self.K, min=0.0)
        payoff = payoff * self.contract_size

        portfolio_value_final = ledger.compute_total_value(S_t, self.N, h_t.mean().item())
        terminal_value = B_t + stock_position * S_t + portfolio_value_final
        terminal_error = terminal_value - self.side * payoff

        trajectories = {
            "S": torch.stack(S_traj, dim=1),
            "B": torch.stack(B_traj, dim=1),
            "stock_positions": torch.stack(stock_pos_traj, dim=1),
            "positions": all_actions,
            "O": None,
            "cost_breakdown": cost_breakdown,
            "soft_constraint_violations": soft_constraint_violations,
            "V0_portfolio": V0_portfolio,
            "ledger_size_trajectory": ledger_size_trajectory,
            "all_actions": all_actions,
        }
        return terminal_error, trajectories


# ----------------------------------------
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
    
    # Compute sparsity penalty (for floating grid mode)
    sparsity_penalty = torch.tensor(0.0, device=terminal_errors.device)
    if lambda_sparsity > 0 and "all_actions" in trajectories:
        all_actions = trajectories["all_actions"]  # [M, N+1, n_instruments]
        option_actions = all_actions[:, :, 1:]  # exclude stock (index 0)
        sparsity_penalty = option_actions.abs().mean()
    
    # Total loss
    total_loss = risk_loss + lambda_constraint * constraint_penalty + lambda_sparsity * sparsity_penalty
    
    return total_loss, risk_loss, constraint_penalty, sparsity_penalty

def train_episode(
    episode: int,
    config: Dict[str, Any],
    policy_net: PolicyNetGARCH,
    optimizer: torch.optim.Optimizer,
    hedged_derivative,
    hedging_derivatives,
    HedgingSim,
    device: torch.device
) -> Dict[str, Any]:
    """Train for a single episode with configurable risk measure and soft constraint."""
    
    hedged_cfg = config["hedged_option"]
    mode = config["instruments"].get("mode", "static")
    is_floating_grid = (mode == "floating_grid")
    
    # Get transaction costs from config
    transaction_costs = get_transaction_costs(config)
    
    # NEW: Get risk measure and soft constraint configuration
    risk_config = config.get("risk_measure", {"type": "mse"})
    constraint_config = config.get("soft_constraint", {"enabled": False, "lambda": 0.0})
    
    risk_measure = risk_config.get("type", "mse")
    alpha = risk_config.get("alpha", None)
    lambda_constraint = constraint_config.get("lambda", 0.0) if constraint_config.get("enabled", False) else 0.0
    
    # Get sparsity penalty for floating grid
    lambda_sparsity = 0.0
    if is_floating_grid:
        lambda_sparsity = config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
    
    # FIX: Convert K to tensor on device BEFORE passing to HedgingSim
    K_tensor = torch.tensor(hedged_cfg["K"], dtype=torch.float32, device=device)
    S0_tensor = torch.tensor(config["simulation"]["S0"], dtype=torch.float32, device=device)
    
    sim = HedgingSim(
        S0=S0_tensor,  # Pass as tensor
        K=K_tensor,    # Pass as tensor
        m=0.1,
        r=config["simulation"]["r"],
        sigma=config["garch"]["sigma0"],
        T=config["simulation"]["T"],
        option_type=hedged_cfg["option_type"],
        position=hedged_cfg["side"],
        M=config["simulation"]["M"],
        N=config["simulation"]["N"],
        TCP=transaction_costs.get('stock', 0.0001),
        seed=episode
    )

    env = HedgingEnvGARCH(
        sim=sim,
        derivative=hedged_derivative,
        hedging_derivatives=None if is_floating else hedging_derivatives,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        transaction_costs=transaction_costs,
        grid_config=config if is_floating else None,
    )

    env.reset()
    S_traj, V_traj, O_traj, obs_sequence, RL_actions = env.simulate_trajectory_and_get_observations(policy_net)
    terminal_errors, trajectories = env.simulate_full_trajectory(RL_actions, O_traj)

    optimizer.zero_grad()
    total_loss, risk_loss, constraint_penalty, sparsity_pen = compute_loss_with_soft_constraint(
        terminal_errors, trajectories, risk_measure=risk_measure, alpha=alpha, lambda_constraint=lambda_constraint, lambda_sparsity=lambda_sparsity
    )
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=config["training"]["gradient_clip_max_norm"])
    optimizer.step()

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logging.error("Loss became NaN/Inf")
        raise RuntimeError("Loss became NaN/Inf")

    final_reward = -float(total_loss.item())

    # Log summary
    log_msg = (
        f"Episode {episode} | Reward: {final_reward:.6f} | "
        f"Total Loss: {total_loss.item():.6f} | Risk Loss: {risk_loss.item():.6f}"
    )
    if lambda_constraint > 0:
        avg_violation = trajectories["soft_constraint_violations"].mean().item()
        log_msg += f" | Constraint: {constraint_penalty.item():.6f} (Avg Viol: {avg_violation:.6f})"
    if lambda_sparsity > 0:
        log_msg += f" | Sparsity: {sparsity_pen.item():.6f}"
    if is_floating and "ledger_size_trajectory" in trajectories:
        avg_ledger = np.mean(trajectories["ledger_size_trajectory"])
        max_ledger = np.max(trajectories["ledger_size_trajectory"])
        log_msg += f" | Ledger (Avg/Max): {avg_ledger:.1f}/{max_ledger:.0f}"
    logging.info(log_msg)

    return {
        "episode": episode,
        "loss": total_loss.item(),
        "risk_loss": risk_loss.item(),
        "constraint_penalty": constraint_penalty.item(),
        "sparsity_penalty": sparsity_pen.item() if isinstance(sparsity_pen, torch.Tensor) else float(sparsity_pen),
        "reward": final_reward,
        "trajectories": trajectories,
        "RL_positions": RL_actions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env,
    }
