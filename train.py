"""
Example training script showing how to use the flexible derivative system.
"""

import yaml
import torch
import argparse
from src.agents.policy_net_garch_flexible import PolicyNetGARCH, HedgingEnvGARCH
from derivative_factory import setup_derivatives_from_precomputed


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # ----------------------------
    # Parse command-line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="Train flexible hedging agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., cfgs/BDGTC.yaml)"
    )
    args = parser.parse_args()

    # ----------------------------
    # Load configuration
    # ----------------------------
    config = load_config(args.config)

    # ----------------------------
    # Setup derivatives (auto-detects vanilla vs barrier)
    # ----------------------------
    hedged_derivative, hedging_derivatives, precomp_managers = setup_derivatives_from_precomputed(config)

    print(f"[INFO] Loaded config from: {args.config}")
    print(f"[INFO] Hedged derivative type: {type(hedged_derivative).__name__}")
    print(f"[INFO] Number of hedging instruments: {len(hedging_derivatives)}")
    for i, deriv in enumerate(hedging_derivatives):
        if deriv is None:
            print(f"  [{i}] Stock")
        else:
            print(f"  [{i}] {type(deriv).__name__}")

    # ----------------------------
    # Simulation config class
    # ----------------------------
    class SimConfig:
        def __init__(self, cfg):
            self.M = cfg['simulation']['M']
            self.N = cfg['simulation']['N']
            self.S0 = cfg['simulation']['S0']
            self.K = cfg['hedged_option']['K']
            self.T = cfg['simulation']['T']
            self.r = cfg['simulation']['r']
            self.sigma = cfg['garch']['sigma0']
            self.side = 1 if cfg['hedged_option']['side'] == 'long' else -1
            self.contract_size = cfg['simulation']['contract_size']
            self.TCP = cfg['simulation']['TCP']
            self.option_type = cfg['hedged_option']['option_type']

    sim = SimConfig(config)

    # ----------------------------
    # Create environment and policy
    # ----------------------------
    env = HedgingEnvGARCH(
        sim=sim,
        derivative=hedged_derivative,
        hedging_derivatives=hedging_derivatives,
        garch_params=config['garch'],
        n_hedging_instruments=config['instruments']['n_hedging_instruments'],
        device=config['training']['device']
    )

    policy_net = PolicyNetGARCH(
        obs_dim=config['model']['obs_dim'],
        hidden_size=config['model']['hidden_size'],
        n_hedging_instruments=config['instruments']['n_hedging_instruments'],
        num_layers=config['model']['num_layers']
    )

    optimizer = torch.optim.AdamW(
        policy_net.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    for episode in range(config['training']['episodes']):
        env.reset()

        # Simulate trajectory
        S_traj, V_traj, O_trajs, obs_seq, positions = env.simulate_trajectory_and_get_observations(policy_net)

        # Compute Greeks
        portfolio_greeks = {
            'delta': env.compute_all_paths_greeks(S_traj, 'delta'),
            'gamma': env.compute_all_paths_greeks(S_traj, 'gamma'),
        }

        if config['instruments']['n_hedging_instruments'] >= 3:
            portfolio_greeks['vega'] = env.compute_all_paths_greeks(S_traj, 'vega')

        if config['instruments']['n_hedging_instruments'] >= 4:
            portfolio_greeks['theta'] = env.compute_all_paths_greeks(S_traj, 'theta')

        # Optimal hedge
        optimal_positions = env.compute_hn_option_positions(S_traj, portfolio_greeks)

        # Loss (MSE)
        loss = torch.mean((positions - optimal_positions) ** 2)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(),
            config['training']['gradient_clip_max_norm']
        )
        optimizer.step()

        if episode % 10 == 0:
            terminal_error, trajectories = env.simulate_full_trajectory(positions, O_trajs)
            pnl_mean = terminal_error.mean().item()
            pnl_std = terminal_error.std().item()
            print(f"Episode {episode}: Loss={loss.item():.6f}, "
                  f"PnL Mean={pnl_mean:.4f}, PnL Std={pnl_std:.4f}")

    # ----------------------------
    # Save model
    # ----------------------------
    torch.save({
        'policy_net': policy_net.state_dict(),
        'config': config
    }, config['output']['model_save_path'])

    print(f"[INFO] Model saved to {config['output']['model_save_path']}")


if __name__ == '__main__':
    main()
