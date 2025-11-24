import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class PolicyNetGARCH(nn.Module):
    def __init__(self, obs_dim=5, hidden_size=128, n_hedging_instruments=2, num_layers=2):
        super().__init__()
        
        self.n_hedging_instruments = n_hedging_instruments
        
        self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers=2, batch_first=True)
        
        self.fc_layers = nn.ModuleList()
        in_dim = hidden_size
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size
        
        self.instrument_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_hedging_instruments)
        ])
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, obs_sequence):
        lstm_out, _ = self.lstm(obs_sequence)
        x = lstm_out
        
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        outputs = [head(x).squeeze(-1) for head in self.instrument_heads]
        
        return outputs


class FloatingGridManager:
    """Manages dynamic option grid generation for floating grid hedging."""
    
    def __init__(self, config: Dict[str, Any], derivative_class, sim_params: Dict[str, Any]):
        """
        Initialize floating grid manager.
        
        Args:
            config: Configuration dict with 'instruments']['floating_grid'] section
            derivative_class: Class to instantiate (e.g., VanillaOption)
            sim_params: Dict with 'r', 'N', 'T', etc.
        """
        grid_config = config["instruments"]["floating_grid"]
        
        self.enabled = grid_config.get("enabled", False)
        if not self.enabled:
            return
        
        self.moneyness_levels = grid_config["moneyness_levels"]
        self.maturity_days = grid_config["maturity_days"]
        self.option_type = grid_config["option_type"]
        self.derivative_class = derivative_class
        self.sim_params = sim_params
        
        self.grid_size = len(self.moneyness_levels) * len(self.maturity_days)
        
        # Validate grid size matches n_hedging_instruments
        expected_size = 1 + self.grid_size  # Stock + option buckets
        actual_size = config["instruments"]["n_hedging_instruments"]
        if expected_size != actual_size:
            raise ValueError(
                f"Grid size mismatch: expected {expected_size} "
                f"(1 stock + {self.grid_size} options), got {actual_size}"
            )
        
        logger.info(
            f"FloatingGridManager initialized: {len(self.moneyness_levels)} moneyness levels × "
            f"{len(self.maturity_days)} maturities = {self.grid_size} option buckets"
        )
    
    def get_bucket_params(self, bucket_idx: int) -> Tuple[float, int]:
        """
        Convert flat bucket index to (moneyness, maturity_days).
        
        Bucket layout: maturity varies fastest
        bucket_0 = (moneyness[0], maturity[0])
        bucket_1 = (moneyness[0], maturity[1])
        ...
        bucket_k = (moneyness[i], maturity[j]) where k = i * len(maturity) + j
        """
        n_maturities = len(self.maturity_days)
        moneyness_idx = bucket_idx // n_maturities
        maturity_idx = bucket_idx % n_maturities
        
        return self.moneyness_levels[moneyness_idx], self.maturity_days[maturity_idx]
    
    def create_derivative(self, S_current: float, bucket_idx: int, 
                         current_step: int, total_steps: int):
        """
        Create derivative for given bucket at current timestep.
        
        Args:
            S_current: Current spot price (scalar)
            bucket_idx: Which bucket (0 to grid_size-1)
            current_step: Current timestep
            total_steps: Total simulation steps
            
        Returns:
            Derivative object
        """
        moneyness, maturity_days = self.get_bucket_params(bucket_idx)
        
        # Strike = moneyness * current spot
        K = moneyness * S_current
        
        # Maturity in timesteps (capped by remaining simulation time)
        days_remaining = total_steps - current_step
        N_option = min(maturity_days, days_remaining)
        
        # Create derivative
        deriv = self.derivative_class(
            K=K,
            N=N_option,
            option_type=self.option_type,
            r=self.sim_params['r'],
            T=self.sim_params['T'],
            S0=S_current,
            M=self.sim_params['M']
        )
        
        return deriv


class PositionLedger:
    """Tracks all historical option positions for floating grid."""
    
    def __init__(self, M: int, device: torch.device):
        """
        Initialize ledger for M paths.
        
        Args:
            M: Number of Monte Carlo paths
            device: torch device
        """
        self.M = M
        self.device = device
        # Each path has its own list of positions
        self.positions = [[] for _ in range(M)]
    
    def add_position(self, derivative, quantity: torch.Tensor, bucket_idx: int,
                    current_step: int, S_current: torch.Tensor):
        """
        Add positions to ledger (per-path).
        
        Args:
            derivative: Option object (same for all paths in bucket)
            quantity: [M] tensor of quantities (can be negative for short)
            bucket_idx: Which bucket this came from
            current_step: Timestep when opened
            S_current: [M] spot prices when opened
        """
        for path_idx in range(self.M):
            qty = float(quantity[path_idx])
            if abs(qty) > 1e-6:  # Only add non-zero positions
                self.positions[path_idx].append({
                    'derivative': derivative,
                    'quantity': qty,
                    'bucket_idx': bucket_idx,
                    'opened_at': current_step,
                    'expiry_step': current_step + derivative.N,
                    'strike': derivative.K,
                    'opening_spot': float(S_current[path_idx])
                })
    
    def remove_expired(self, current_step: int):
        """Remove positions that have expired."""
        for path_idx in range(self.M):
            self.positions[path_idx] = [
                pos for pos in self.positions[path_idx]
                if pos['expiry_step'] > current_step
            ]
    
    def compute_and_remove_expired(self, S_current: torch.Tensor, 
                                   current_step: int, h0: float) -> torch.Tensor:
        """
        Compute payoffs for expired positions, remove them, return payoffs.
        
        Args:
            S_current: [M] current spot prices
            current_step: Current timestep
            h0: GARCH variance
            
        Returns:
            payoffs: [M] total payoffs from expired positions
        """
        payoffs = torch.zeros(self.M, device=self.device)
        
        for path_idx in range(self.M):
            expired = []
            remaining = []
            
            for pos in self.positions[path_idx]:
                if pos['expiry_step'] <= current_step:
                    expired.append(pos)
                else:
                    remaining.append(pos)
            
            # Compute payoffs for expired
            for pos in expired:
                S_path = S_current[path_idx].item()
                K = pos['strike']
                opt_type = pos['derivative'].option_type.lower()
                
                if opt_type == 'call':
                    payoff = max(S_path - K, 0.0)
                else:  # put
                    payoff = max(K - S_path, 0.0)
                
                payoffs[path_idx] += pos['quantity'] * payoff
            
            self.positions[path_idx] = remaining
        
        return payoffs
    
    def compute_total_value(self, S_current: torch.Tensor, 
                           current_step: int, h0: float) -> torch.Tensor:
        """
        Price all active positions in ledger.
        
        Args:
            S_current: [M] current spot prices
            current_step: Current timestep
            h0: GARCH variance
            
        Returns:
            portfolio_value: [M] total portfolio value
        """
        portfolio_value = torch.zeros(self.M, device=self.device)
        
        for path_idx in range(self.M):
            for pos in self.positions[path_idx]:
                if pos['expiry_step'] > current_step:
                    # Price this position
                    S_path = S_current[path_idx].unsqueeze(0)
                    
                    price = pos['derivative'].price(
                        S=S_path,
                        K=pos['strike'],
                        step_idx=current_step,
                        N=pos['derivative'].N,
                        h0=h0
                    )
                    
                    portfolio_value[path_idx] += pos['quantity'] * price.item()
        
        return portfolio_value
    
    def compute_greeks(self, S_current: torch.Tensor, current_step: int,
                      h0: float, greek_name: str) -> torch.Tensor:
        """
        Compute aggregate Greek across all positions.
        
        Args:
            S_current: [M] spot prices
            current_step: Current timestep
            h0: GARCH variance
            greek_name: 'delta', 'gamma', 'vega', or 'theta'
            
        Returns:
            greeks: [M] aggregate Greek values
        """
        greeks = torch.zeros(self.M, device=self.device)
        
        for path_idx in range(self.M):
            for pos in self.positions[path_idx]:
                if pos['expiry_step'] > current_step:
                    S_path = S_current[path_idx].unsqueeze(0)
                    
                    greek_method = getattr(pos['derivative'], greek_name)
                    greek_val = greek_method(
                        S=S_path,
                        K=pos['strike'],
                        step_idx=current_step,
                        N=pos['derivative'].N,
                        h0=h0
                    )
                    
                    greeks[path_idx] += pos['quantity'] * greek_val.item()
        
        return greeks
    
    def get_ledger_size(self) -> torch.Tensor:
        """
        Get number of active positions per path.
        
        Returns:
            sizes: [M] number of positions per path
        """
        sizes = torch.tensor(
            [len(self.positions[path_idx]) for path_idx in range(self.M)],
            dtype=torch.float32,
            device=self.device
        )
        return sizes


class HedgingEnvGARCH:
    def __init__(self, sim, derivative, hedging_derivatives: Optional[List],
                 garch_params=None, n_hedging_instruments=2,
                 dt_min=1e-10, device="cpu", transaction_costs=None,
                 grid_config=None):
        """
        Initialize hedging environment with static or floating grid support.
        
        Args:
            sim: Simulation parameters object
            derivative: The derivative to hedge
            hedging_derivatives: List of derivative objects (for static mode) or None (for floating)
            garch_params: GARCH model parameters
            n_hedging_instruments: Number of hedging instruments (stock + options)
            dt_min: Minimum time step
            device: 'cpu' or 'cuda'
            transaction_costs: Dict with keys 'stock', 'vanilla_option', etc.
            grid_config: Dict with floating grid configuration (if applicable)
        """
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)
        
        self.derivative = derivative
        self.n_hedging_instruments = n_hedging_instruments
        
        # Detect mode
        self.is_floating_grid = (grid_config is not None and 
                                grid_config.get("instruments", {}).get("floating_grid", {}).get("enabled", False))
        
        if self.is_floating_grid:
            # Floating grid mode
            from src.option_greek.vanilla import VanillaOption
            
            self.grid_manager = FloatingGridManager(
                config=grid_config,
                derivative_class=VanillaOption,
                sim_params={
                    'r': sim.r / 252.0,
                    'T': sim.T,
                    'N': sim.N,
                    'M': sim.M
                }
            )
            self.position_ledger = None  # Will be created per episode
            self.hedging_derivatives_static = None
            
            # All grid options are vanilla
            self.instrument_types_list = ['stock'] + ['vanilla_option'] * self.grid_manager.grid_size
            
            # Observation dimension increases
            self.obs_dim = 9  # time, moneyness, stock_pos, V/S, side*V, ledger_size, portfolio_val, delta, gamma
            
            # Sparsity penalty
            self.sparsity_penalty = grid_config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
            self.min_trade_size = grid_config["instruments"]["floating_grid"].get("min_trade_size", 0.1)
            
            logger.info(f"Floating grid mode: {self.grid_manager.grid_size} option buckets, obs_dim={self.obs_dim}")
        else:
            # Static grid mode
            self.grid_manager = None
            self.position_ledger = None
            self.hedging_derivatives_static = hedging_derivatives
            self.instrument_types_list = self._classify_instruments()
            self.obs_dim = 5
            self.sparsity_penalty = 0.0
            self.min_trade_size = 0.0
            
            logger.info(f"Static grid mode: {n_hedging_instruments} instruments, obs_dim={self.obs_dim}")
        
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
        
        self.S0 = self.sim.S0
        self.K = self.sim.K
        self.T = self.sim.T
        self.r = self.sim.r / 252.0
        self.option_type = getattr(self.sim, 'option_type', 'call')
        self.side = self.sim.side
        self.contract_size = self.sim.contract_size
        
        self.transaction_costs = transaction_costs or {
            'stock': getattr(self.sim, "TCP", 0.0001),
            'vanilla_option': getattr(self.sim, "TCP", 0.0001) * 10,
            'barrier_option': getattr(self.sim, "TCP", 0.0001) * 20,
            'american_option': getattr(self.sim, "TCP", 0.0001) * 15,
        }
        
        # Static mode: extract instrument metadata
        if not self.is_floating_grid:
            self.instrument_maturities = []
            self.instrument_strikes = []
            self.instrument_types = []
            
            for deriv in self.hedging_derivatives_static:
                if deriv is None:
                    self.instrument_maturities.append(0)
                    self.instrument_strikes.append(0)
                    self.instrument_types.append('stock')
                else:
                    self.instrument_maturities.append(getattr(deriv, 'N', self.N))
                    self.instrument_strikes.append(getattr(deriv, 'K', self.K))
                    
                    from src.option_greek.vanilla import VanillaOption
                    from src.option_greek.barrier import BarrierOption
                    from src.option_greek.american import AmericanOption
                    
                    if isinstance(deriv, VanillaOption):
                        self.instrument_types.append('vanilla')
                    elif isinstance(deriv, BarrierOption):
                        self.instrument_types.append('barrier')
                    elif isinstance(deriv, AmericanOption):
                        self.instrument_types.append('american')
                    else:
                        self.instrument_types.append('vanilla')
        
        self.sigma0 = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), self.sigma0, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)
    
    def _classify_instruments(self) -> List[str]:
        """Classify each hedging instrument (static mode only)."""
        from src.option_greek.vanilla import VanillaOption
        from src.option_greek.barrier import BarrierOption
        from src.option_greek.american import AmericanOption
        
        instrument_types = []
        
        for deriv in self.hedging_derivatives_static:
            if deriv is None:
                instrument_types.append('stock')
            elif isinstance(deriv, AmericanOption):
                instrument_types.append('american_option')
            elif isinstance(deriv, BarrierOption):
                instrument_types.append('barrier_option')
            else:
                instrument_types.append('vanilla_option')
        
        return instrument_types
    
    def _get_transaction_cost_rate(self, instrument_idx: int) -> float:
        """Get transaction cost rate for instrument."""
        instrument_type = self.instrument_types_list[instrument_idx]
        return self.transaction_costs.get(instrument_type, 0.0001)
    
    def reset(self):
        """Reset environment state."""
        self.Z = torch.randn((self.M, self.N), dtype=torch.float32, device=self.device)
        sigma0_annual = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), sigma0_annual, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)
        
        # Reset ledger for floating grid
        if self.is_floating_grid:
            self.position_ledger = PositionLedger(self.M, self.device)
        
        # Reset barrier status if applicable
        from src.option_greek.barrier_wrapper import BarrierOptionWithVanillaFallback
        
        if isinstance(self.derivative, BarrierOptionWithVanillaFallback):
            self.derivative.reset_barrier_status()
        
        if not self.is_floating_grid:
            for deriv in self.hedging_derivatives_static:
                if isinstance(deriv, BarrierOptionWithVanillaFallback):
                    deriv.reset_barrier_status()
    
    def simulate_trajectory_and_get_observations(self, policy_net):
        """
        Simulate trajectory using LSTM.
        
        Returns:
            S_trajectory: [M, N+1] stock prices
            V_trajectory: [M, N+1] hedged derivative values
            O_trajectories_or_None: List of [M, N+1] (static) or None (floating)
            obs_sequence: [M, N+1, obs_dim] observations
            all_actions: [M, N+1, n_instruments] actions
        """
        if self.is_floating_grid:
            return self._simulate_floating_grid(policy_net)
        else:
            return self._simulate_static_grid(policy_net)
    
    def _simulate_static_grid(self, policy_net):
        """Original static grid simulation."""
        S_trajectory = []
        V_trajectory = []
        O_trajectories = [[] for _ in range(len(self.hedging_derivatives_static) - 1)]
        obs_list = []
        
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        
        S_trajectory.append(S_t)
        
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)
        
        for i, hedge_deriv in enumerate(self.hedging_derivatives_static[1:]):
            K_hedge = getattr(hedge_deriv, 'K', self.K)
            N_hedge = getattr(hedge_deriv, 'N', self.N)
            O0 = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=0, N=N_hedge, h0=h0_current)
            O_trajectories[i].append(O0)
        
        obs_t = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0
        obs_t[:, 0, 1] = S_t / self.K
        obs_t[:, 0, 2] = 0.5
        obs_t[:, 0, 3] = V0 / S_t
        obs_t[:, 0, 4] = self.side * V0
        obs_list.append(obs_t)
        
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))
        
        outputs = []
        for i, head in enumerate(policy_net.instrument_heads):
            output = head(x).squeeze(-1)[:, 0]
            outputs.append(output)
        
        positions_t = torch.stack(outputs, dim=-1)
        all_positions = [positions_t]
        
        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            h0_current = h_t.mean().item()
            V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            
            for i, hedge_deriv in enumerate(self.hedging_derivatives_static[1:]):
                K_hedge = getattr(hedge_deriv, 'K', self.K)
                N_hedge = getattr(hedge_deriv, 'N', self.N)
                O_t = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=t+1, N=N_hedge, h0=h0_current)
                O_trajectories[i].append(O_t)
            
            S_trajectory.append(S_t)
            V_trajectory.append(V_t)
            
            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t / self.K
            obs_new[:, 0, 2] = positions_t[:, 0].detach()
            obs_new[:, 0, 3] = V_t / S_t
            obs_new[:, 0, 4] = self.side * V_t
            obs_list.append(obs_new)
            
            lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
            x = lstm_out
            for fc in policy_net.fc_layers:
                x = F.relu(fc(x))
            
            outputs = []
            for i, head in enumerate(policy_net.instrument_heads):
                output = head(x).squeeze(-1)[:, 0]
                outputs.append(output)
            
            positions_t = torch.stack(outputs, dim=-1)
            all_positions.append(positions_t)
        
        S_trajectory = torch.stack(S_trajectory, dim=1)
        V_trajectory = torch.stack(V_trajectory, dim=1)
        O_trajectories = [torch.stack(traj, dim=1) for traj in O_trajectories]
        all_positions = torch.stack(all_positions, dim=1)
        obs_sequence = torch.cat(obs_list, dim=1)
        
        return S_trajectory, V_trajectory, O_trajectories, obs_sequence, all_positions
    
    def _simulate_floating_grid(self, policy_net):
        """Floating grid simulation with ledger tracking."""
        S_trajectory = []
        V_trajectory = []
        obs_list = []
        all_actions = []
        
        # Initialize ledger
        ledger = PositionLedger(self.M, self.device)
        
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        stock_position = torch.zeros(self.M, dtype=torch.float32, device=self.device)
        
        S_trajectory.append(S_t)
        
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)
        
        # Initial observation (9 dims, all ledger stats are zero)
        obs_t = torch.zeros((self.M, 1, 9), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0  # time
        obs_t[:, 0, 1] = S_t / self.K  # moneyness
        obs_t[:, 0, 2] = 0.0  # stock position
        obs_t[:, 0, 3] = V0 / S_t  # V/S
        obs_t[:, 0, 4] = self.side * V0  # side*V
        obs_t[:, 0, 5] = 0.0  # ledger size
        obs_t[:, 0, 6] = 0.0  # portfolio value
        obs_t[:, 0, 7] = 0.0  # delta
        obs_t[:, 0, 8] = 0.0  # gamma
        obs_list.append(obs_t)
        
        # Get initial actions
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))
        
        outputs = []
        for i, head in enumerate(policy_net.instrument_heads):
            output = head(x).squeeze(-1)[:, 0]
            outputs.append(output)
        
        actions_t = torch.stack(outputs, dim=-1)
        all_actions.append(actions_t)
        
        # Main loop
        for t in range(self.N):
            # Execute previous actions
            actions_prev = all_actions[-1]
            
            # Stock: target position (same as before)
            stock_trade = actions_prev[:, 0] - stock_position
            stock_position = actions_prev[:, 0]
            
            # Options: incremental trades
            for bucket_idx in range(1, self.n_hedging_instruments):
                trade_qty = torch.round(actions_prev[:, bucket_idx])  # Integerize per-path
                
                # Filter by min_trade_size
                trade_qty = torch.where(
                    trade_qty.abs() < self.min_trade_size,
                    torch.zeros_like(trade_qty),
                    trade_qty
                )
                
                if trade_qty.abs().sum() > 1e-6:  # Any non-zero trades
                    # Create derivative for this bucket
                    S_mean = S_t.mean().item()
                    deriv = self.grid_manager.create_derivative(
                        S_current=S_mean,
                        bucket_idx=bucket_idx - 1,  # Adjust for stock at index 0
                        current_step=t,
                        total_steps=self.N
                    )
                    
                    # Add to ledger
                    ledger.add_position(deriv, trade_qty, bucket_idx, t, S_t)
            
            # Evolve state
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            # Remove expired positions
            ledger.remove_expired(t + 1)
            
            # Price current state
            h0_current = h_t.mean().item()
V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            
            # Price ledger and compute Greeks
            portfolio_value = ledger.compute_total_value(S_t, t+1, h0_current)
            total_delta = ledger.compute_greeks(S_t, t+1, h0_current, 'delta')
            total_gamma = ledger.compute_greeks(S_t, t+1, h0_current, 'gamma')
            ledger_sizes = ledger.get_ledger_size()
            
            S_trajectory.append(S_t)
            V_trajectory.append(V_t)
            
            # Create observation (9 dims)
            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, 9), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t / self.K
            obs_new[:, 0, 2] = stock_position.detach()
            obs_new[:, 0, 3] = V_t / S_t
            obs_new[:, 0, 4] = self.side * V_t
            obs_new[:, 0, 5] = ledger_sizes / 100.0  # Normalize (assume max 100 positions)
            obs_new[:, 0, 6] = portfolio_value / (V_t + 1e-8)  # Normalize
            obs_new[:, 0, 7] = total_delta
            obs_new[:, 0, 8] = total_gamma
            obs_list.append(obs_new)
            
            # Get next actions (if not at terminal)
            if t < self.N - 1:
                lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
                x = lstm_out
                for fc in policy_net.fc_layers:
                    x = F.relu(fc(x))
                
                outputs = []
                for i, head in enumerate(policy_net.instrument_heads):
                    output = head(x).squeeze(-1)[:, 0]
                    outputs.append(output)
                
                actions_t = torch.stack(outputs, dim=-1)
                all_actions.append(actions_t)
        
        S_trajectory = torch.stack(S_trajectory, dim=1)
        V_trajectory = torch.stack(V_trajectory, dim=1)
        all_actions = torch.stack(all_actions, dim=1)
        obs_sequence = torch.cat(obs_list, dim=1)
        
        # Store ledger for second simulation
        self.position_ledger = ledger
        
        return S_trajectory, V_trajectory, None, obs_sequence, all_actions
    
    def simulate_full_trajectory(self, all_actions, O_trajectories_or_None):
        """
        Simulate full hedging trajectory with transaction costs and P&L.
        
        Args:
            all_actions: [M, N+1, n_instruments] actions (positions for static, trades for floating)
            O_trajectories_or_None: List of [M, N+1] (static) or None (floating)
            
        Returns:
            terminal_error: [M] terminal hedging errors
            trajectories: Dict with trajectories and metrics
        """
        if self.is_floating_grid:
            return self._simulate_full_trajectory_floating(all_actions)
        else:
            return self._simulate_full_trajectory_static(all_actions, O_trajectories_or_None)
    
    def _simulate_full_trajectory_static(self, all_positions, O_trajectories):
        """Original static grid P&L calculation."""
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        positions_t = all_positions[:, 0]
        
        h0_current = self.h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        
        # Initialize bank account
        B_t = self.side * V0 - positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            B_t -= positions_t[:, i+1] * O_trajectories[i][:, 0]
        
        V0_portfolio = self.side * V0
        h_t = self.h_t.clone()
        
        S_traj, B_traj = [S_t], [B_t]
        
        cost_breakdown = {
            'stock': torch.zeros(self.M, device=self.device),
            'vanilla_option': torch.zeros(self.M, device=self.device),
            'barrier_option': torch.zeros(self.M, device=self.device),
            'american_option': torch.zeros(self.M, device=self.device),
        }
        
        soft_constraint_violations = torch.zeros(self.M, device=self.device)
        
        for t in range(self.N):
            # Evolve GARCH
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            
            # Evolve stock
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            positions_new = all_positions[:, t+1]
            
            # Accrue interest
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)
            
            # Rebalance
            for i in range(self.n_hedging_instruments):
                trade = positions_new[:, i] - positions_t[:, i]
                tcp_rate = self._get_transaction_cost_rate(i)
                
                if i == 0:
                    price = S_t
                    instrument_type = 'stock'
                else:
                    price = O_trajectories[i-1][:, t+1]
                    instrument_type = self.instrument_types_list[i]
                
                cost = tcp_rate * torch.abs(trade) * price
                B_t = B_t - trade * price - cost
                cost_breakdown[instrument_type] += cost
            
            positions_t = positions_new
            
            # Portfolio value
            P_t = B_t + positions_t[:, 0] * S_t
            for i in range(len(O_trajectories)):
                P_t += positions_t[:, i+1] * O_trajectories[i][:, t+1]
            
            # Soft constraint tracking
            h0_current = h_t.mean().item()
            V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            V_t_phi = self.side * V_t_phi
            xi_t = P_t - V_t_phi
            soft_constraint_violations += torch.clamp(xi_t, min=0.0)
            
            S_traj.append(S_t)
            B_traj.append(B_t)
        
        # Terminal payoff
        if hasattr(self.derivative, 'option_type'):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            payoff = torch.clamp(S_t - self.K, min=0.0)
        
        payoff = payoff * self.contract_size
        
        # Terminal portfolio value
        terminal_value = B_t + positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            terminal_value += positions_t[:, i+1] * O_trajectories[i][:, -1]
        
        terminal_error = terminal_value - self.side * payoff
        
        trajectories = {
            'S': torch.stack(S_traj, dim=1),
            'B': torch.stack(B_traj, dim=1),
            'positions': all_positions,
            'O': O_trajectories,
            'cost_breakdown': cost_breakdown,
            'soft_constraint_violations': soft_constraint_violations,
            'V0_portfolio': V0_portfolio
        }
        
        return terminal_error, trajectories
    
    def _simulate_full_trajectory_floating(self, all_actions):
        """Floating grid P&L calculation with ledger."""
        # Rebuild ledger from actions
        ledger = PositionLedger(self.M, self.device)
        
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        stock_position = torch.zeros(self.M, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        
        # Initialize bank account with derivative proceeds
        B_t = self.side * V0
        
        V0_portfolio = self.side * V0
        
        S_traj, B_traj = [S_t], [B_t]
        stock_pos_traj = [stock_position.clone()]
        
        cost_breakdown = {
            'stock': torch.zeros(self.M, device=self.device),
            'vanilla_option': torch.zeros(self.M, device=self.device),
        }
        
        soft_constraint_violations = torch.zeros(self.M, device=self.device)
        ledger_size_trajectory = [0]
        
        for t in range(self.N):
            # Execute actions from timestep t
            actions_t = all_actions[:, t]
            
            # Stock: target position semantics
            stock_trade = actions_t[:, 0] - stock_position
            tcp_stock = self._get_transaction_cost_rate(0)
            stock_cost = tcp_stock * torch.abs(stock_trade) * S_t
            B_t -= stock_trade * S_t + stock_cost
            cost_breakdown['stock'] += stock_cost
            stock_position = actions_t[:, 0]
            
            # Options: incremental trade semantics
            tcp_option = self._get_transaction_cost_rate(1)  # All grid options same type
            
            for bucket_idx in range(1, self.n_hedging_instruments):
                trade_qty = torch.round(actions_t[:, bucket_idx])
                
                # Filter by min_trade_size
                trade_qty = torch.where(
                    trade_qty.abs() < self.min_trade_size,
                    torch.zeros_like(trade_qty),
                    trade_qty
                )
                
                if trade_qty.abs().sum() > 1e-6:
                    # Create derivative
                    S_mean = S_t.mean().item()
                    deriv = self.grid_manager.create_derivative(
                        S_current=S_mean,
                        bucket_idx=bucket_idx - 1,
                        current_step=t,
                        total_steps=self.N
                    )
                    
                    # Price it for transaction cost
                    option_prices = torch.zeros(self.M, device=self.device)
                    for path_idx in range(self.M):
                        if abs(trade_qty[path_idx]) > 1e-6:
                            S_path = S_t[path_idx].unsqueeze(0)
                            price = deriv.price(S=S_path, K=deriv.K, step_idx=t, N=deriv.N, h0=h0_current)
                            option_prices[path_idx] = price.item()
                    
                    # Transaction costs
                    option_cost = tcp_option * trade_qty.abs() * option_prices
                    B_t -= trade_qty * option_prices + option_cost
                    cost_breakdown['vanilla_option'] += option_cost
                    
                    # Add to ledger
                    ledger.add_position(deriv, trade_qty, bucket_idx, t, S_t)
            
            # Evolve state
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            # Accrue interest
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)
            
            # Handle expirations
            expired_payoffs = ledger.compute_and_remove_expired(S_t, t+1, h_t.mean().item())
            B_t += expired_payoffs
            
            # Compute portfolio value
            h0_current = h_t.mean().item()
            portfolio_value = ledger.compute_total_value(S_t, t+1, h0_current)
            P_t = B_t + stock_position * S_t + portfolio_value
            
            # Soft constraint tracking
            V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            V_t_phi = self.side * V_t_phi
            xi_t = P_t - V_t_phi
            soft_constraint_violations += torch.clamp(xi_t, min=0.0)
            
            S_traj.append(S_t)
            B_traj.append(B_t)
            stock_pos_traj.append(stock_position.clone())
            ledger_size_trajectory.append(ledger.get_ledger_size().mean().item())
        
        # Terminal payoff
        if hasattr(self.derivative, 'option_type'):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            payoff = torch.clamp(S_t - self.K, min=0.0)
        
        payoff = payoff * self.contract_size
        
        # Terminal portfolio value
        portfolio_value_final = ledger.compute_total_value(S_t, self.N, h_t.mean().item())
        terminal_value = B_t + stock_position * S_t + portfolio_value_final
        
        terminal_error = terminal_value - self.side * payoff
        
        trajectories = {
            'S': torch.stack(S_traj, dim=1),
            'B': torch.stack(B_traj, dim=1),
            'stock_positions': torch.stack(stock_pos_traj, dim=1),
            'positions': all_actions,  # For compatibility
            'O': None,
            'cost_breakdown': cost_breakdown,
            'soft_constraint_violations': soft_constraint_violations,
            'V0_portfolio': V0_portfolio,
            'ledger_size_trajectory': ledger_size_trajectory,
            'all_actions': all_actions  # Store for sparsity penalty
        }
        
        return terminal_error, trajectories


def compute_loss_with_soft_constraint(terminal_error, trajectories, risk_measure='mse', 
                                     alpha=None, lambda_constraint=0.0, lambda_sparsity=0.0):
    """
    Compute loss function with soft constraint and sparsity penalty.
    
    Args:
        terminal_error: [M] terminal hedging errors
        trajectories: Dict containing 'soft_constraint_violations' and optionally 'all_actions'
        risk_measure: 'mse', 'smse', 'cvar', 'var', or 'mae'
        alpha: Confidence level for CVaR/VaR
        lambda_constraint: Weight for soft constraint penalty
        lambda_sparsity: Weight for sparsity penalty (floating grid only)
    
    Returns:
        total_loss: Combined loss
        risk_loss: Risk measure component
        constraint_penalty: Soft constraint component
        sparsity_penalty: Sparsity component (0 if not applicable)
    """
    M = terminal_error.shape[0]
    
    # Compute risk measure
    if risk_measure == 'mse':
        risk_loss = (terminal_error ** 2).mean()
    elif risk_measure == 'smse':
        positive_mask = (terminal_error >= 0).float()
        risk_loss = ((terminal_error ** 2) * positive_mask).mean()
    elif risk_measure == 'cvar':
        if alpha is None:
            raise ValueError("alpha parameter required for CVaR")
        sorted_errors, _ = torch.sort(terminal_error, descending=True)
        n_tail = max(1, int(np.ceil(M * (1 - alpha))))
        risk_loss = sorted_errors[:n_tail].mean()
    elif risk_measure == 'var':
        if alpha is None:
            raise ValueError("alpha parameter required for VaR")
        risk_loss = torch.quantile(terminal_error, alpha)
    elif risk_measure == 'mae':
        risk_loss = terminal_error.abs().mean()
    else:
        raise ValueError(f"Unknown risk measure: {risk_measure}")
    
    # Soft constraint penalty
    soft_violations = trajectories['soft_constraint_violations']
    constraint_penalty = soft_violations.mean()
    
    # Sparsity penalty (only for floating grid)
    sparsity_penalty = torch.tensor(0.0, device=terminal_error.device)
    if lambda_sparsity > 0 and 'all_actions' in trajectories:
        all_actions = trajectories['all_actions']  # [M, N+1, n_instruments]
        # Only penalize option actions (exclude stock at index 0)
        option_actions = all_actions[:, :, 1:]
        sparsity_penalty = option_actions.abs().mean()
    
    # Combined loss
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
    """Train for a single episode."""
    
    hedged_cfg = config["hedged_option"]
    
    # Detect mode
    mode = config["instruments"].get("mode", "static")
    is_floating = (mode == "floating_grid")
    
    # Get transaction costs
    from train import get_transaction_costs  # Assuming this exists
    transaction_costs = get_transaction_costs(config)
    
    # Get risk measure config
    risk_config = config.get("risk_measure", {"type": "mse"})
    constraint_config = config.get("soft_constraint", {"enabled": False, "lambda": 0.0})
    
    risk_measure = risk_config.get("type", "mse")
    alpha = risk_config.get("alpha", None)
    lambda_constraint = constraint_config.get("lambda", 0.0) if constraint_config.get("enabled", False) else 0.0
    
    # Get sparsity penalty (floating grid only)
    lambda_sparsity = 0.0
    if is_floating:
        lambda_sparsity = config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
    
    sim = HedgingSim(
        S0=config["simulation"]["S0"],
        K=hedged_cfg["K"],
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
        hedging_derivatives=hedging_derivatives if not is_floating else None,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        transaction_costs=transaction_costs,
        grid_config=config if is_floating else None
    )
    
    env.reset()
    
    S_traj, V_traj, O_traj, obs_sequence, RL_actions = \
        env.simulate_trajectory_and_get_observations(policy_net)
    
    terminal_errors, trajectories = env.simulate_full_trajectory(RL_actions, O_traj)
    
    optimizer.zero_grad()
    
    total_loss, risk_loss, constraint_penalty, sparsity_pen = compute_loss_with_soft_constraint(
        terminal_errors, 
        trajectories,
        risk_measure=risk_measure,
        alpha=alpha,
        lambda_constraint=lambda_constraint,
        lambda_sparsity=lambda_sparsity
    )
    
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=config["training"]["gradient_clip_max_norm"]
    )
    
    optimizer.step()
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logging.error("Loss became NaN/Inf")
        raise RuntimeError("Loss became NaN/Inf")
    
    final_reward = -float(total_loss.item())
    
    # Logging
    log_msg = (
        f"Episode {episode} | Reward: {final_reward:.6f} | "
        f"Total Loss: {total_loss.item():.6f} | Risk Loss: {risk_loss.item():.6f}"
    )
    
    if lambda_constraint > 0:
        avg_violation = trajectories['soft_constraint_violations'].mean().item()
        log_msg += f" | Constraint: {constraint_penalty.item():.6f} (Avg Viol: {avg_violation:.6f})"
    
    if lambda_sparsity > 0:
        log_msg += f" | Sparsity: {sparsity_pen.item():.6f}"
    
    if is_floating and 'ledger_size_trajectory' in trajectories:
        avg_ledger = np.mean(trajectories['ledger_size_trajectory'])
        max_ledger = np.max(trajectories['ledger_size_trajectory'])
        log_msg += f" | Ledger (Avg/Max): {avg_ledger:.1f}/{max_ledger:.0f}"
    
    logging.info(log_msg)
    
    return {
        "episode": episode,
        "loss": total_loss.item(),
        "risk_loss": risk_loss.item(),
        "constraint_penalty": constraint_penalty.item(),
        "sparsity_penalty": sparsity_pen.item(),
        "reward": final_reward,
        "trajectories": trajectories,
        "RL_positions": RL_actions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env
    }
