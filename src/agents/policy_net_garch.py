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

        # LSTM to process the observation sequence
        self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers=2, batch_first=True)

        # Create multiple FC layers dynamically
        self.fc_layers = nn.ModuleList()
        in_dim = hidden_size
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size

        # Create output heads dynamically for each instrument
        self.instrument_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_hedging_instruments)
        ])

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, obs_sequence):
        lstm_out, _ = self.lstm(obs_sequence)
        x = lstm_out

        # Pass through all FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        # Get positions for all instruments
        outputs = [head(x).squeeze(-1) for head in self.instrument_heads]

        return outputs


# --------------------------------------------------------------------------
# Hedging Environment (Batched Trajectory Simulation)
# --------------------------------------------------------------------------
class HedgingEnvGARCH:
    def __init__(self, sim, derivative, hedging_derivatives: List,
                 garch_params=None, n_hedging_instruments=2,
                 dt_min=1e-10, device="cpu"):
        """
        Initialize hedging environment with flexible derivative support.
        
        Args:
            sim: Simulation parameters object
            derivative: The derivative to hedge (VanillaOption or BarrierOption)
            hedging_derivatives: List of derivative objects to use for hedging
            garch_params: GARCH model parameters
            n_hedging_instruments: Number of hedging instruments (stock + options)
            dt_min: Minimum time step
            device: 'cpu' or 'cuda'
        """
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)

        # Store the derivative to hedge
        self.derivative = derivative
        
        # Store hedging derivatives (first is stock, rest are options)
        self.hedging_derivatives = hedging_derivatives
        self.n_hedging_instruments = n_hedging_instruments

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
        self.TCP = getattr(self.sim, "TCP", 0.0)

        self.sigma0 = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), self.sigma0, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)

    def reset(self):
        """Reset environment state."""
        self.Z = torch.randn((self.M, self.N), dtype=torch.float32, device=self.device)
        sigma0_annual = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), sigma0_annual, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)

    def compute_all_paths_greeks(self, S_trajectory, greek_name: str):
        """
        Compute specified Greek for the hedged derivative across all paths.
        
        Args:
            S_trajectory: [M, N+1] stock prices
            greek_name: 'delta', 'gamma', 'vega', or 'theta'
        
        Returns:
            [M, N+1] tensor of Greek values
        """
        M, N_plus_1 = S_trajectory.shape
        greek_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)

        greek_method = getattr(self.derivative, greek_name)

        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            h0_current = self.h_t.mean().item()  # Use mean variance across paths
            
            # Call the Greek method - handles both vanilla and barrier options
            greek_t = greek_method(S=S_t, K=self.K, step_idx=t, N=self.N, h0=h0_current)
            greek_trajectory[:, t] = greek_t

        return greek_trajectory

    def compute_hn_option_positions(self, S_trajectory, portfolio_greeks):
        """
        Compute optimal hedge positions using linear algebra for n Greeks with n instruments.

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

        # Define which greeks to hedge based on n
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

        # Compute greeks for all hedging instruments
        instrument_greeks = []

        for j, hedge_deriv in enumerate(self.hedging_derivatives):
            if hedge_deriv is None:  # Stock
                greeks = {
                    'delta': torch.ones((M, N_plus_1), device=self.device),
                    'gamma': torch.zeros((M, N_plus_1), device=self.device),
                    'vega': torch.zeros((M, N_plus_1), device=self.device),
                    'theta': torch.zeros((M, N_plus_1), device=self.device)
                }
            else:  # Option (vanilla or barrier)
                greeks = {}
                h0_mean = self.h_t.mean().item()
                
                for greek_name in ['delta', 'gamma', 'vega', 'theta']:
                    greek_traj = torch.zeros((M, N_plus_1), device=self.device)
                    greek_method = getattr(hedge_deriv, greek_name)
                    
                    for t in range(N_plus_1):
                        S_t = S_trajectory[:, t]
                        # Get the strike from the derivative
                        K_hedge = getattr(hedge_deriv, 'K', self.K)
                        N_hedge = getattr(hedge_deriv, 'N', self.N)
                        
                        greek_traj[:, t] = greek_method(
                            S=S_t, K=K_hedge, step_idx=t, N=N_hedge, h0=h0_mean
                        )
                    
                    greeks[greek_name] = greek_traj

            instrument_greeks.append(greeks)

        # Build A matrix: [M, N+1, n, n]
        A = torch.zeros((M, N_plus_1, n, n), device=self.device)

        for i, greek_name in enumerate(greek_names):
            for j, inst_greeks in enumerate(instrument_greeks):
                A[:, :, i, j] = inst_greeks[greek_name]

        # Build b vector: [M, N+1, n]
        b = torch.stack([-portfolio_greeks[g] for g in greek_names], dim=-1)

        # Solve using ridge regularization for stability
        epsilon = 1e-12
        lambda_reg = 1e-6

        # Row-normalize
        row_norm = A.norm(dim=-1, keepdim=True)
        A_scaled = A / (row_norm + epsilon)
        b_scaled = b / (row_norm[..., 0] + epsilon)

        # Ridge least-squares
        I = torch.eye(n, device=A.device).reshape(1, 1, n, n)
        ATA = torch.matmul(A_scaled.transpose(-2, -1), A_scaled)
        ATb = torch.matmul(A_scaled.transpose(-2, -1), b_scaled.unsqueeze(-1))

        x = torch.linalg.solve(ATA + lambda_reg * I, ATb)
        x = x.squeeze(-1)

        return x

    def simulate_trajectory_and_get_observations(self, policy_net):
        """
        Simulate trajectory using LSTM with proper hidden state management.

        Returns:
            S_trajectory: [M, N+1] stock prices
            V_trajectory: [M, N+1] hedged derivative values
            O_trajectories: List of [M, N+1] tensors for hedging derivative prices
            obs_sequence: [M, N+1, 5] observations
            all_positions: [M, N+1, n_instruments] positions
        """
        S_trajectory = []
        V_trajectory = []
        O_trajectories = [[] for _ in range(len(self.hedging_derivatives) - 1)]  # Exclude stock
        obs_list = []

        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()

        S_trajectory.append(S_t)

        # Price hedged derivative
        h0_current = h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)
        V_trajectory.append(V0)

        # Price all hedging derivatives
        for i, hedge_deriv in enumerate(self.hedging_derivatives[1:]):  # Skip stock
            K_hedge = getattr(hedge_deriv, 'K', self.K)
            N_hedge = getattr(hedge_deriv, 'N', self.N)
            O0 = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=0, N=N_hedge, h0=h0_current)
            O_trajectories[i].append(O0)

        # Initial observation
        obs_t = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0
        obs_t[:, 0, 1] = S_t / self.K
        obs_t[:, 0, 2] = 0.5
        obs_t[:, 0, 3] = V0 / S_t
        obs_t[:, 0, 4] = self.side * V0
        obs_list.append(obs_t)

        # Get initial positions
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))

        outputs = []
        for i, head in enumerate(policy_net.instrument_heads):
            if i == 0:
                output = torch.sigmoid(head(x)).squeeze(-1)[:, 0]
            else:
                output = head(x).squeeze(-1)[:, 0]
            outputs.append(output)

        positions_t = torch.stack(outputs, dim=-1)
        all_positions = [positions_t]

        # Simulate trajectory
        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            h0_current = h_t.mean().item()

            # Price hedged derivative
            V_t = self.derivative.price(S=S_t, K=self.K, step_idx=t+1, N=self.N, h0=h0_current)

            # Price all hedging derivatives
            for i, hedge_deriv in enumerate(self.hedging_derivatives[1:]):
                K_hedge = getattr(hedge_deriv, 'K', self.K)
                N_hedge = getattr(hedge_deriv, 'N', self.N)
                O_t = hedge_deriv.price(S=S_t, K=K_hedge, step_idx=t+1, N=N_hedge, h0=h0_current)
                O_trajectories[i].append(O_t)

            S_trajectory.append(S_t)
            V_trajectory.append(V_t)

            # Create observation
            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t / self.K
            obs_new[:, 0, 2] = positions_t[:, 0].detach()
            obs_new[:, 0, 3] = V_t / S_t
            obs_new[:, 0, 4] = self.side * V_t
            obs_list.append(obs_new)

            # Get new positions
            lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
            x = lstm_out
            for fc in policy_net.fc_layers:
                x = F.relu(fc(x))

            outputs = []
            for i, head in enumerate(policy_net.instrument_heads):
                if i == 0:
                    output = torch.sigmoid(head(x)).squeeze(-1)[:, 0]
                else:
                    output = head(x).squeeze(-1)[:, 0]
                outputs.append(output)

            positions_t = torch.stack(outputs, dim=-1)
            all_positions.append(positions_t)

        S_trajectory = torch.stack(S_trajectory, dim=1)
        V_trajectory = torch.stack(V_trajectory, dim=1)

        # Stack option trajectories
        O_trajectories = [torch.stack(traj, dim=1) for traj in O_trajectories]

        all_positions = torch.stack(all_positions, dim=1)
        obs_sequence = torch.cat(obs_list, dim=1)

        return S_trajectory, V_trajectory, O_trajectories, obs_sequence, all_positions

    def simulate_full_trajectory(self, all_positions, O_trajectories):
        """
        Simulate full hedging trajectory with n instruments.

        Args:
            all_positions: [M, N+1, n_instruments] positions for all instruments
            O_trajectories: List of [M, N+1] tensors for hedging option prices
        """
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        positions_t = all_positions[:, 0]

        h0_current = self.h_t.mean().item()
        V0 = self.derivative.price(S=S_t, K=self.K, step_idx=0, N=self.N, h0=h0_current)

        # Initialize bank account
        B_t = self.side * V0 - positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            B_t -= positions_t[:, i+1] * O_trajectories[i][:, 0]

        h_t = self.h_t.clone()

        S_traj, B_traj = [S_t], [B_t]
        position_trajs = {i: [positions_t[:, i]] for i in range(self.n_hedging_instruments)}

        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            positions_new = all_positions[:, t+1]

            # Update bank account
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)

            # Transaction costs for all instruments
            for i in range(self.n_hedging_instruments):
                trade = positions_new[:, i] - positions_t[:, i]

                if i == 0:  # Stock
                    price = S_t
                    multiplier = 1
                else:  # Options
                    price = O_trajectories[i-1][:, t+1]
                    multiplier = 10

                cost = self.TCP * multiplier * torch.abs(trade) * price
                B_t = B_t - trade * price - cost

            positions_t = positions_new

            S_traj.append(S_t)
            B_traj.append(B_t)
            for i in range(self.n_hedging_instruments):
                position_trajs[i].append(positions_t[:, i])

        # Terminal payoff
        if hasattr(self.derivative, 'option_type'):
            opt_type = self.derivative.option_type.lower()
            if opt_type == "call":
                payoff = torch.clamp(S_t - self.K, min=0.0)
            else:
                payoff = torch.clamp(self.K - S_t, min=0.0)
        else:
            # Barrier option - check if activated
            payoff = torch.clamp(S_t - self.K, min=0.0)  # Default to call

        payoff = payoff * self.contract_size

        # Terminal value
        terminal_value = B_t + positions_t[:, 0] * S_t
        for i in range(len(O_trajectories)):
            terminal_value += positions_t[:, i+1] * O_trajectories[i][:, -1]

        terminal_error = terminal_value - self.side * payoff

        trajectories = {
            'S': torch.stack(S_traj, dim=1),
            'B': torch.stack(B_traj, dim=1),
            'positions': all_positions,
            'O': O_trajectories,
        }

        return terminal_error, trajectories
