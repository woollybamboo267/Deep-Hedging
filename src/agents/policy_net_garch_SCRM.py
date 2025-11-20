import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, List

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


class HedgingEnvGARCH:
    def __init__(self, sim, derivative, hedging_derivatives: List,
                 garch_params=None, n_hedging_instruments=2,
                 dt_min=1e-10, device="cpu", transaction_costs=None):
        """
        Initialize hedging environment with flexible derivative support.
        
        Args:
            sim: Simulation parameters object
            derivative: The derivative to hedge (VanillaOption, BarrierOption, AmericanOption, or AsianOption)
            hedging_derivatives: List of derivative objects to use for hedging
            garch_params: GARCH model parameters
            n_hedging_instruments: Number of hedging instruments (stock + options)
            dt_min: Minimum time step
            device: 'cpu' or 'cuda'
            transaction_costs: Dict with keys 'stock', 'vanilla_option', 'barrier_option', 'american_option', 'asian_option'
        """
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)
        
        self.derivative = derivative
        self.hedging_derivatives = hedging_derivatives
        self.n_hedging_instruments = n_hedging_instruments
        
        # Check if main derivative is Asian (needs running average)
        from src.option_greek.asian import AsianOption
        self.is_asian_hedged = isinstance(derivative, AsianOption)
        
        # Check which hedging instruments are Asian
        self.hedging_is_asian = [isinstance(d, AsianOption) if d is not None else False 
                                  for d in hedging_derivatives]
        
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
            'asian_option': getattr(self.sim, "TCP", 0.0001) * 15
        }
        
        self.instrument_types_list = self._classify_instruments()
        
        self.instrument_maturities = []
        self.instrument_strikes = []
        self.instrument_types = []
        
        for deriv in self.hedging_derivatives:
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
                from src.option_greek.asian import AsianOption
                
                if isinstance(deriv, VanillaOption):
                    self.instrument_types.append('vanilla')
                elif isinstance(deriv, BarrierOption):
                    self.instrument_types.append('barrier')
                elif isinstance(deriv, AmericanOption):
                    self.instrument_types.append('american')
                elif isinstance(deriv, AsianOption):
                    self.instrument_types.append('asian')
                else:
                    self.instrument_types.append('vanilla')
        
        self.sigma0 = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), self.sigma0, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)
        
        # Initialize running averages for Asian options
        self.A_hedged = None
        self.A_hedging = None
    
    def _classify_instruments(self) -> List[str]:
        """
        Classify each hedging instrument as 'stock', 'vanilla_option', 'barrier_option', 'american_option', or 'asian_option'.
        
        Returns:
            List of instrument type strings
        """
        from src.option_greek.vanilla import VanillaOption
        from src.option_greek.barrier import BarrierOption
        from src.option_greek.american import AmericanOption
        from src.option_greek.asian import AsianOption
        
        instrument_types = []
        
        for deriv in self.hedging_derivatives:
            if deriv is None:
                instrument_types.append('stock')
            elif isinstance(deriv, AmericanOption):
                instrument_types.append('american_option')
            elif isinstance(deriv, BarrierOption):
                instrument_types.append('barrier_option')
            elif isinstance(deriv, AsianOption):
                instrument_types.append('asian_option')
            else:
                instrument_types.append('vanilla_option')
        
        return instrument_types
    
    def _get_transaction_cost_rate(self, instrument_idx: int) -> float:
        """
        Get the transaction cost rate for a specific instrument.
        
        Args:
            instrument_idx: Index of the instrument (0 = stock, 1+ = options)
            
        Returns:
            Transaction cost rate as a float
        """
        instrument_type = self.instrument_types_list[instrument_idx]
        return self.transaction_costs.get(instrument_type, 0.0001)
    
    def _update_running_average(self, A_prev, S_current, t, N):
        """
        Update running average for Asian option.
        
        Arithmetic average formula: A_t = (t * A_{t-1} + S_t) / (t + 1)
        
        Args:
            A_prev: Previous average [M] or None
            S_current: Current spot price [M]
            t: Current timestep (0-indexed)
            N: Total timesteps
        
        Returns:
            Updated average [M]
        """
        if A_prev is None:
            return S_current.clone()
        else:
            return (t * A_prev + S_current) / (t + 1)
    
    def reset(self):
        """Reset environment state including running averages."""
        self.Z = torch.randn((self.M, self.N), dtype=torch.float32, device=self.device)
        sigma0_annual = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), sigma0_annual, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)
        
        # Reset running averages
        self.A_hedged = None
        self.A_hedging = [None] * len([d for d in self.hedging_derivatives if d is not None])
        
        # Reset barrier breach status for wrapped barrier options
        from src.option_greek.barrier_wrapper import BarrierOptionWithVanillaFallback
        
        if isinstance(self.derivative, BarrierOptionWithVanillaFallback):
            self.derivative.reset_barrier_status()
        
        for deriv in self.hedging_derivatives:
            if isinstance(deriv, BarrierOptionWithVanillaFallback):
                deriv.reset_barrier_status()
    
    def compute_all_paths_greeks(self, S_trajectory, greek_name: str):
        """
        Compute specified Greek for the hedged derivative across all paths.
        NOW SUPPORTS ASIAN OPTIONS with running average tracking.
        
        Args:
            S_trajectory: [M, N+1] stock prices
            greek_name: 'delta', 'gamma', 'vega', or 'theta'
        
        Returns:
            [M, N+1] tensor of Greek values
        """
        M, N_plus_1 = S_trajectory.shape
        greek_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)
        
        greek_method = getattr(self.derivative, greek_name)
        
        A_t = None
        
        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            h0_current = self.h_t.mean().item()
            
            if self.is_asian_hedged:
                A_t = self._update_running_average(A_t, S_t, t, self.N)
                greek_t = greek_method(S=S_t, K=self.K, step_idx=t, N=self.N, h0=h0_current, A=A_t)
            else:
                greek_t = greek_method(S=S_t, K=self.K, step_idx=t, N=self.N, h0=h0_current)
            
            greek_trajectory[:, t] = greek_t
        
        return greek_trajectory
    
    def compute_hn_option_positions(self, S_trajectory, portfolio_greeks):
        """
        Compute optimal hedge positions using linear algebra for n Greeks with n instruments.
        NOW SUPPORTS ASIAN HEDGING INSTRUMENTS.
        
        Solves the system:
            A @ x = b
        where A[i,j] = greek_i of instrument_j
        
        Args:
            S_trajectory: [M, N+1] stock prices
            portfolio_greeks: dict with keys like 'delta', 'gamma', 'vega', 'theta' containing [M, N+1] tensors
        
        Returns:
            positions: [M, N+1, n_instruments] positions for each instrument
        """
        M, N_plus_1 = S_trajectory.shape
        n = self.n_hedging_instruments
        epsilon = 1e-10
        
        if n == 1:
            greek_names = ['delta']
        elif n == 2:
            greek_names = ['delta', 'gamma']
        elif n == 3:
            greek_names = ['delta', 'gamma', 'vega']
        elif n == 4:
            greek_names = ['delta', 'gamma', 'vega', 'theta']
        else:
            raise ValueError(f"n_hedging_instruments must be 1-4, got {n}")
        
        instrument_greeks = []
        
        A_hedging = [None] * len(self.hedging_derivatives)
        
        for j, hedge_deriv in enumerate(self.hedging_derivatives):
            if hedge_deriv is None:
                greeks = {
                    'delta': torch.ones((M, N_plus_1), device=self.device),
                    'gamma': torch.zeros((M, N_plus_1), device=self.device),
                    'vega': torch.zeros((M, N_plus_1), device=self.device),
                    'theta': torch.zeros((M, N_plus_1), device=self.device)
                }
            else:
                greeks = {}
                h0_mean = self.h_t.mean().item()
                
                is_asian_hedge = self.hedging_is_asian[j]
                
                for greek_name in ['delta', 'gamma', 'vega', 'theta']:
                    greek_traj = torch.zeros((M, N_plus_1), device=self.device)
                    greek_method = getattr(hedge_deriv, greek_name)
                    
                    A_t = None
                    
                    for t in range(N_plus_1):
                        S_t = S_trajectory[:, t]
                        K_hedge = getattr(hedge_deriv, 'K', self.K)
                        N_hedge = getattr(hedge_deriv, 'N', self.N)
                        
                        if is_asian_hedge:
                            A_t = self._update_running_average(A_t, S_t, t, N_hedge)
                            greek_traj[:, t] = greek_method(
                                S=S_t, K=K_hedge, step_idx=t, N=N_hedge, h0=h0_mean, A=A_t
                            )
                        else:
                            greek_traj[:, t] = greek_method(
                                S=S_t, K=K_hedge, step_idx=t, N=N_hedge, h0=h0_mean
                            )
                    
                    greeks[greek_name] = greek_traj
            
            instrument_greeks.append(greeks)
        
        A = torch.zeros((M, N_plus_1, n, n), device=self.device)
        
        for i, greek_name in enumerate(greek_names):
            for j, inst_greeks in enumerate(instrument_greeks):
                A[:, :, i, j] = inst_greeks[greek_name]
        
        b = torch.stack([-portfolio_greeks[g] for g in greek_names], dim=-1)
        
        epsilon = 1e-12
        lambda_reg = 1e-6
        
        row_norm = A.norm(dim=-1, keepdim=True)
        A_scaled = A / (row_norm + epsilon)
        b_scaled = b / (row_norm[..., 0] + epsilon)
        
        I = torch.eye(n, device=A.device).reshape(1, 1, n, n)
        ATA = torch.matmul(A_scaled.transpose(-2, -1), A_scaled)
        ATb = torch.matmul(A_scaled.transpose(-2, -1), b_scaled.unsqueeze(-1))
        
        x = torch.linalg.solve(ATA + lambda_reg * I, ATb)
        x = x.squeeze(-1)
        
        return x
    
    def simulate_trajectory_and_get_observations(self, policy_net):
        """
        Simulate trajectory using LSTM with proper hidden state management.
        NOW TRACKS RUNNING AVERAGES FOR ASIAN OPTIONS.
        
        Returns:
            S_trajectory: [M, N+1] stock prices
            V_trajectory: [M, N+1] hedged derivative values
            O_trajectories: List of [M, N+1] tensors for hedging derivative prices
            obs_sequence: [M, N+1, 5] observations
            all_positions: [M, N+1, n_instruments] positions
        """
        S_trajectory = []
        V_trajectory = []
        O_trajectories = [[] for _ in range(len(self.hedging_derivatives) - 1)]
        obs_list = []
        
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()
        
        A_hedged = S_t.clone() if self.is_asian_hedged else None
        A_hedging = [S_t.clone() if is_asian else None 
                     for is_asian in self.hedging_is_asian[1:]]
        
        S_trajectory.append(S_t)
        
        h0_current = h_t.mean().item()
        
        if self.is_asian_hedged:
            V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current, A=A_hedged)
        else:
            V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)
        
        for i, hedge_deriv in enumerate(self.hedging_derivatives[1:]):
            K_hedge = getattr(hedge_deriv, 'K', self.K)
            N_hedge = getattr(hedge_deriv, 'N', self.N)
            
            if self.hedging_is_asian[i+1]:
                O0 = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=0, N=N_hedge, h0=h0_current, A=A_hedging[i])
            else:
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
            
            if self.is_asian_hedged:
                A_hedged = self._update_running_average(A_hedged, S_t, t+1, self.N)
            
            for i in range(len(A_hedging)):
                if self.hedging_is_asian[i+1]:
                    A_hedging[i] = self._update_running_average(A_hedging[i], S_t, t+1, self.N)
            
            h0_current = h_t.mean().item()
            
            if self.is_asian_hedged:
                V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current, A=A_hedged)
            else:
                V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            
            for i, hedge_deriv in enumerate(self.hedging_derivatives[1:]):
                K_hedge = getattr(hedge_deriv, 'K', self.K)
                N_hedge = getattr(hedge_deriv, 'N', self.N)
                
                if self.hedging_is_asian[i+1]:
                    O_t = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=t+1, N=N_hedge, h0=h0_current, A=A_hedging[i])
                else:
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
    
    def simulate_full_trajectory(self, all_positions, O_trajectories):
        """
        Simulate full hedging trajectory with n instruments and instrument-specific costs.
        NOW HANDLES ASIAN OPTIONS IN TERMINAL PAYOFF CALCULATION.
        INCLUDES SOFT ERROR CONSTRAINT TRACKING.
        
        Args:
            all_positions: [M, N+1, n_instruments] positions for all instruments
            O_trajectories: List of [M, N+1] tensors for hedging option prices
            
        Returns:
            terminal_error: Terminal hedging error [M]
            trajectories: Dict containing trajectories and soft constraint violation
        """
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        positions_t = all_positions[:, 0]
        
        h0_current = self.h_t.mean().item()
        
        A_hedged = S_t.clone() if self.is_asian_hedged else None
        
        if self.is_asian_hedged:
            V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current, A=A_hedged)
        else:
            V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        
        # Initialize bank account: short derivative, hold hedge positions
        B_t = self.side * V0 - positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            B_t -= positions_t[:, i+1] * O_trajectories[i][:, 0]
        
        # Store initial portfolio value V0 for soft constraint
        V0_portfolio = self.side * V0
        
        h_t = self.h_t.clone()
        
        S_traj, B_traj = [S_t], [B_t]
        position_trajs = {i: [positions_t[:, i]] for i in range(self.n_hedging_instruments)}
        
        cost_breakdown = {
            'stock': torch.zeros(self.M, device=self.device),
            'vanilla_option': torch.zeros(self.M, device=self.device),
            'barrier_option': torch.zeros(self.M, device=self.device),
            'american_option': torch.zeros(self.M, device=self.device),
            'asian_option': torch.zeros(self.M, device=self.device)
        }
        
        # Initialize soft constraint tracking: ξ_t^ϕθ = P_t - V_t^ϕθ
        # We accumulate max(0, ξ_t^ϕθ) across all timesteps
        soft_constraint_violations = torch.zeros(self.M, device=self.device)
        
        for t in range(self.N):
            # Evolve GARCH volatility
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)
            
            # Evolve stock price
            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)
            
            # Update running average for Asian at timestep t+1
            if self.is_asian_hedged:
                A_hedged = self._update_running_average(A_hedged, S_t, t+1, self.N)
            
            positions_new = all_positions[:, t+1]
            
            # Accrue interest on bank account
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)
            
            # Rebalance each instrument with appropriate transaction cost
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
            
            # Calculate current portfolio value: P_t = B_t + sum(positions_i * prices_i)
            P_t = B_t + positions_t[:, 0] * S_t
            for i in range(len(O_trajectories)):
                P_t += positions_t[:, i+1] * O_trajectories[i][:, t+1]
            
            # Price the derivative at current timestep for comparison
            h0_current = h_t.mean().item()
            if self.is_asian_hedged:
                V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current, A=A_hedged)
            else:
                V_t_phi = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)
            V_t_phi = self.side * V_t_phi
            
            # Compute tracking error: ξ_t = P_t - V_t^ϕθ
            xi_t = P_t - V_t_phi
            
            # Accumulate positive tracking errors (violations): max(0, ξ_t)
            # This penalizes when portfolio value exceeds derivative value
            soft_constraint_violations += torch.clamp(xi_t, min=0.0)
            
            S_traj.append(S_t)
            B_traj.append(B_t)
            for i in range(self.n_hedging_instruments):
                position_trajs[i].append(positions_t[:, i])
        
        # Compute terminal payoff based on derivative type
        if self.is_asian_hedged:
            if hasattr(self.derivative, 'option_type'):
                opt_type = self.derivative.option_type.lower()
                if opt_type == "call":
                    payoff = torch.clamp(A_hedged - self.K, min=0.0)
                else:  # put
                    payoff = torch.clamp(self.K - A_hedged, min=0.0)
            else:
                payoff = torch.clamp(A_hedged - self.K, min=0.0)
        else:
            if hasattr(self.derivative, 'option_type'):
                opt_type = self.derivative.option_type.lower()
                if opt_type == "call":
                    payoff = torch.clamp(S_t - self.K, min=0.0)
                else:  # put
                    payoff = torch.clamp(self.K - S_t, min=0.0)
            else:
                payoff = torch.clamp(S_t - self.K, min=0.0)
        
        payoff = payoff * self.contract_size
        
        # Calculate terminal portfolio value
        terminal_value = B_t + positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            terminal_value += positions_t[:, i+1] * O_trajectories[i][:, -1]
        
        # Terminal hedging error
        terminal_error = terminal_value - self.side * payoff
        
        trajectories = {
            'S': torch.stack(S_traj, dim=1),
            'B': torch.stack(B_traj, dim=1),
            'positions': all_positions,
            'O': O_trajectories,
            'cost_breakdown': cost_breakdown,
            'soft_constraint_violations': soft_constraint_violations,  # NEW: accumulated violations
            'V0_portfolio': V0_portfolio  # NEW: initial portfolio value for reference
        }
        
        return terminal_error, trajectories


def compute_loss_with_soft_constraint(terminal_error, trajectories, risk_measure='cvar', alpha=0.95, lambda_constraint=1.0):
    """
    Compute loss function with soft error constraint.
    
    Following equation (6): O(θ; λ) = ρ(ξ_T^ϕθ) + λ·SC(θ)
    
    where SC(θ) = P(max_{t∈{0,...,T}} {ξ_t^ϕθ} > V_0)
    
    Risk measures implemented:
    - MSE: ρ(ξ_T^ϕ) = E[(ξ_T^ϕ)²]
    - SMSE: ρ(ξ_T^ϕ) = E[(ξ_T^ϕ)² · 𝟙_{ξ_T^ϕ ≥ 0}]
    - CVaR_α: ρ(ξ_T^ϕ) = E[ξ_T^ϕ | ξ_T^ϕ ≥ VaR_α(ξ_T^ϕ)]
    
    Args:
        terminal_error: [M] terminal hedging errors (ξ_T^ϕθ)
        trajectories: Dict containing 'soft_constraint_violations' [M]
        risk_measure: 'mse', 'smse', 'cvar', 'variance', or 'mae'
        alpha: CVaR confidence level (default: 0.95, meaning we look at worst 5%)
        lambda_constraint: Weight for soft constraint penalty (default: 1.0)
    
    Returns:
        total_loss: Combined loss (risk measure + soft constraint penalty)
        risk_loss: Risk measure component only
        constraint_penalty: Soft constraint penalty component only
    """
    M = terminal_error.shape[0]
    
    # Compute primary risk measure ρ(ξ_T^ϕθ)
    if risk_measure == 'mse':
        # Mean Square Error: E[(ξ_T^ϕ)²]
        risk_loss = (terminal_error ** 2).mean()
    
    elif risk_measure == 'smse':
        # Semi Mean-Square Error: E[(ξ_T^ϕ)² · 𝟙_{ξ_T^ϕ ≥ 0}]
        # Only penalizes positive errors (losses from hedger's perspective)
        positive_mask = (terminal_error >= 0).float()
        risk_loss = ((terminal_error ** 2) * positive_mask).mean()
    
    elif risk_measure == 'cvar':
        # Conditional Value-at-Risk (CVaR_α):
        # E[ξ_T^ϕ | ξ_T^ϕ ≥ VaR_α(ξ_T^ϕ)]
        # 
        # VaR_α is the α-quantile: min{c : P(ξ_T^ϕ ≤ c) ≥ α}
        # CVaR_α is the expected value in the tail beyond VaR_α
        #
        # For α=0.95, we compute the mean of the worst 5% of errors
        
        # Sort errors in descending order (worst first)
        sorted_errors, _ = torch.sort(terminal_error, descending=True)
        
        # Number of samples in the (1-α) tail (worst 5% for α=0.95)
        n_tail = int(np.ceil(M * (1 - alpha)))
        n_tail = max(1, n_tail)  # At least 1 sample
        
        # CVaR is the mean of the worst (1-α)% samples
        risk_loss = sorted_errors[:n_tail].mean()
    
    elif risk_measure == 'variance':
        # Variance of terminal error: Var(ξ_T^ϕ)
        risk_loss = terminal_error.var()
    
    elif risk_measure == 'mae':
        # Mean Absolute Error: E[|ξ_T^ϕ|]
        risk_loss = terminal_error.abs().mean()
    
    else:
        raise ValueError(f"Unknown risk measure: {risk_measure}. Choose from: 'mse', 'smse', 'cvar', 'variance', 'mae'")
    
    # Compute soft constraint penalty: SC(θ)
    # We use the accumulated violations normalized by number of paths
    soft_violations = trajectories['soft_constraint_violations']  # [M]
    
    # Average accumulated violation across all paths
    constraint_penalty = soft_violations.mean()
    
    # Alternative: probability of any violation occurring
    # constraint_penalty = (soft_violations > 0).float().mean()
    
    # Combined objective: O(θ; λ) = ρ(ξ_T^ϕθ) + λ·SC(θ)
    total_loss = risk_loss + lambda_constraint * constraint_penalty
    
    return total_loss, risk_loss, constraint_penalty


# Example usage in training loop:
"""
def train_step(policy_net, env, optimizer, risk_measure='cvar', alpha=0.95, lambda_constraint=1.0):
    '''
    Single training step with soft constraint.
    
    Args:
        policy_net: Policy network
        env: HedgingEnvGARCH environment
        optimizer: PyTorch optimizer
        risk_measure: Risk measure to use
        alpha: CVaR confidence level
        lambda_constraint: Weight for soft constraint (higher = stronger enforcement)
    '''
    optimizer.zero_grad()
    
    # Simulate trajectory
    S_trajectory, V_trajectory, O_trajectories, obs_sequence, all_positions = \
        env.simulate_trajectory_and_get_observations(policy_net)
    
    # Compute terminal error and trajectories (includes soft constraint tracking)
    terminal_error, trajectories = env.simulate_full_trajectory(all_positions, O_trajectories)
    
    # Compute loss with soft constraint
    total_loss, risk_loss, constraint_penalty = compute_loss_with_soft_constraint(
        terminal_error, trajectories, 
        risk_measure=risk_measure, 
        alpha=alpha, 
        lambda_constraint=lambda_constraint
    )
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'risk_loss': risk_loss.item(),
        'constraint_penalty': constraint_penalty.item(),
        'terminal_error_mean': terminal_error.mean().item(),
        'terminal_error_std': terminal_error.std().item(),
        'avg_soft_violation': trajectories['soft_constraint_violations'].mean().item()
    }
"""
