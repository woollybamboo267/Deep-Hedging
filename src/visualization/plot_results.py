import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from typing import Dict, Any, Tuple
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_practitioner_benchmark(
    env: Any,
    S_traj: torch.Tensor,
    O_traj: Dict[int, torch.Tensor],
    n_hedging_instruments: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], np.ndarray]:
    """Compute the practitioner benchmark hedge."""
    S_np = S_traj.cpu().numpy()
    
    # Determine Greeks to hedge
    if n_hedging_instruments == 1:
        greek_names = ['delta']
    elif n_hedging_instruments == 2:
        greek_names = ['delta', 'gamma']
    elif n_hedging_instruments == 3:
        greek_names = ['delta', 'gamma', 'vega']
    elif n_hedging_instruments == 4:
        greek_names = ['delta', 'gamma', 'vega', 'theta']
    else:
        raise ValueError(f"n_hedging_instruments must be 1-4, got {n_hedging_instruments}")
    
    # Compute portfolio Greeks
    portfolio_greeks = {}
    for greek_name in greek_names:
        greek_traj = env.compute_all_paths_greeks(S_traj, greek_name)
        portfolio_greeks[greek_name] = -env.side * greek_traj
    
    # Solve for optimal hedge positions
    HN_positions_all = env.compute_hn_option_positions(S_traj, portfolio_greeks)
    
    # Simulate hedge strategy
    _, trajectories_hn = env.simulate_full_trajectory(HN_positions_all, O_traj)
    
    # Compute terminal hedge errors
    S_final = trajectories_hn['S'][:, -1]
    
    # Compute payoff
    if hasattr(env.derivative, 'option_type'):
        opt_type = env.derivative.option_type.lower()
    else:
        opt_type = env.option_type.lower()
    
    if opt_type == "call":
        payoff = torch.clamp(S_final - env.K, min=0.0)
    else:
        payoff = torch.clamp(env.K - S_final, min=0.0)
    payoff = payoff * env.contract_size
    
    terminal_value_hn = trajectories_hn['B'][:, -1] + HN_positions_all[:, -1, 0] * S_final
    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
        option_contrib = HN_positions_all[:, -1, i] * O_traj[i - 1][:, -1]
        terminal_value_hn += option_contrib
    
    terminal_hedge_error_hn = (terminal_value_hn - env.side * payoff).cpu().detach().numpy()
    
    return HN_positions_all, trajectories_hn, terminal_hedge_error_hn


def compute_rl_metrics(
    env: Any,
    RL_positions: torch.Tensor,
    trajectories: Dict[str, torch.Tensor],
    O_traj: Dict[int, torch.Tensor]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute RL strategy metrics."""
    S_final = trajectories['S'][:, -1]
    
    # Determine option type from derivative
    if hasattr(env.derivative, 'option_type'):
        opt_type = env.derivative.option_type.lower()
    else:
        opt_type = env.option_type.lower()
    
    if opt_type == "call":
        payoff = torch.clamp(S_final - env.K, min=0.0)
    else:
        payoff = torch.clamp(env.K - S_final, min=0.0)
    
    payoff = payoff * env.contract_size
    
    # Compute terminal value
    terminal_value_rl = trajectories['B'][:, -1] + RL_positions[:, -1, 0] * S_final
    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
        terminal_value_rl += RL_positions[:, -1, i] * O_traj[i - 1][:, -1]
    
    terminal_hedge_error_rl = (terminal_value_rl - env.side * payoff).cpu().detach().numpy()
    
    # Compute metrics
    mse_rl = float(np.mean(terminal_hedge_error_rl ** 2))
    smse_rl = mse_rl / (env.S0 ** 2)
    cvar_95_rl = float(np.mean(np.sort(terminal_hedge_error_rl ** 2)[-int(0.05 * env.M):]))
    
    metrics = {
        'mse': mse_rl,
        'smse': smse_rl,
        'cvar_95': cvar_95_rl
    }
    
    return terminal_hedge_error_rl, metrics


def plot_episode_results(
    episode: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Create comprehensive visualization of episode results."""
    try:
        logger.info(f"Generating plots for episode {episode}")
        
        # Extract data from metrics
        env = metrics['env']
        RL_positions = metrics['RL_positions']
        trajectories = metrics['trajectories']
        S_traj = metrics['S_traj']
        V_traj = metrics['V_traj']
        O_traj = metrics['O_traj']
        
        n_inst = config["instruments"]["n_hedging_instruments"]
        path_idx = config["output"]["sample_path_index"]
        
        derivative_type = type(env.derivative).__name__
        
        # Convert O_traj to dict if it's a list
        if isinstance(O_traj, list):
            O_traj = {i: tensor for i, tensor in enumerate(O_traj)}
        
        # Compute RL metrics
        terminal_hedge_error_rl, rl_metrics = compute_rl_metrics(
            env, RL_positions, trajectories, O_traj
        )
        
        # Compute practitioner benchmark
        HN_positions_all, trajectories_hn, terminal_hedge_error_hn = \
            compute_practitioner_benchmark(env, S_traj, O_traj, n_inst)
        
        # Compute HN metrics
        mse_hn = float(np.mean(terminal_hedge_error_hn ** 2))
        smse_hn = mse_hn / (env.S0 ** 2)
        cvar_95_hn = float(np.mean(np.sort(terminal_hedge_error_hn ** 2)[-int(0.05 * env.M):]))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        time_steps = np.arange(env.N + 1)
        
        # Extract sample paths for plotting
        rl_positions_sample = RL_positions[path_idx].cpu().detach().numpy()
        hn_positions_sample = HN_positions_all[path_idx].cpu().detach().numpy()
        
        # PLOT 1: Stock Delta Comparison
        axes[0, 0].plot(time_steps, rl_positions_sample[:, 0], label='RL Delta',
                        linewidth=2, color='tab:blue')
        axes[0, 0].plot(time_steps, hn_positions_sample[:, 0], label='Practitioner Delta',
                        linewidth=2, linestyle='--', alpha=0.8, color='tab:orange')
        axes[0, 0].set_xlabel("Time Step", fontsize=11)
        axes[0, 0].set_ylabel("Delta", fontsize=11)
        axes[0, 0].set_title(f"Stock Delta: Practitioner vs RL (Path {path_idx})", fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # PLOT 2: Option Positions Comparison
        if n_inst >= 2:
            for i in range(1, n_inst):
                maturity = env.instrument_maturities[i]
                opt_type = env.instrument_types[i]
                strike = env.instrument_strikes[i]
                label_suffix = f'{maturity}d {opt_type.upper()} K={strike}'
                
                axes[0, 1].plot(time_steps, rl_positions_sample[:, i],
                              label=f'RL {label_suffix}', linewidth=2)
                axes[0, 1].plot(time_steps, hn_positions_sample[:, i],
                              label=f'Prac {label_suffix}', linewidth=2,
                              linestyle='--', alpha=0.8)
            axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[0, 1].set_xlabel("Time Step", fontsize=11)
            axes[0, 1].set_ylabel("Option Contracts", fontsize=11)
            axes[0, 1].set_title(f"Option Positions: Practitioner vs RL (Path {path_idx})", fontsize=12)
            axes[0, 1].legend(fontsize=9)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No option positions\n(Delta hedge only)',
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Option Positions", fontsize=12)
        
        # PLOT 3: Stock Price Trajectory with Derivative Price
        ax1 = axes[1, 0]
        color_stock = 'tab:green'
        ax1.plot(time_steps, S_traj[path_idx].cpu().detach().numpy(),
                 label='Stock Price', color=color_stock, linewidth=2)
        ax1.axhline(y=env.K, color='r', linestyle='--', label='Strike', alpha=0.7)
        
        # Add special features based on derivative type
        if hasattr(env.derivative, 'barrier_level'):
            # Barrier option - add barrier level
            barrier_val = env.derivative.barrier_level
            if isinstance(barrier_val, torch.Tensor):
                barrier_val = barrier_val.item()
            ax1.axhline(y=barrier_val, color='purple', 
                       linestyle=':', label='Barrier', alpha=0.7, linewidth=2)
        
        # For Asian options, you could add average price line if needed
        # if derivative_type == 'AsianOption':
        #     running_avg = S_traj[path_idx].cpu().detach().numpy().cumsum() / np.arange(1, len(time_steps) + 1)
        #     ax1.plot(time_steps, running_avg, color='cyan', 
        #              linestyle='-.', label='Running Avg', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel("Time Step", fontsize=11)
        ax1.set_ylabel("Stock Price", fontsize=11, color=color_stock)
        ax1.tick_params(axis='y', labelcolor=color_stock)
        ax1.grid(True, alpha=0.3)
        
        # Plot derivative price on secondary y-axis
        ax2 = ax1.twinx()
        color_derivative = 'tab:blue'
        ax2.plot(time_steps, V_traj[path_idx].cpu().detach().numpy(),
                 label=f'{derivative_type} Price', color=color_derivative, 
                 linewidth=2, alpha=0.8)
        ax2.set_ylabel(f"{derivative_type} Price", fontsize=11, color=color_derivative)
        ax2.tick_params(axis='y', labelcolor=color_derivative)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')
        
        ax1.set_title(f"Stock & {derivative_type} Price (Path {path_idx})", fontsize=12)
        
        # PLOT 4: Hedging Instrument Prices
        if n_inst >= 2:
            for i in range(1, n_inst):
                opt_idx = i - 1
                if opt_idx not in O_traj:
                    continue
                    
                maturity = env.instrument_maturities[i]
                opt_type = env.instrument_types[i]
                strike = env.instrument_strikes[i]
                
                option_prices = O_traj[opt_idx][path_idx].cpu().detach().numpy()
                axes[1, 1].plot(time_steps, option_prices,
                              label=f'{maturity}d {opt_type.upper()} K={strike}', linewidth=2)
        
        axes[1, 1].set_xlabel("Time Step", fontsize=11)
        axes[1, 1].set_ylabel("Option Price", fontsize=11)
        axes[1, 1].set_title(f"Hedging Instrument Prices (Path {path_idx})", fontsize=12)
        if n_inst >= 2:
            axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # PLOT 5: Position Difference (RL - Practitioner)
        delta_diff = rl_positions_sample[:, 0] - hn_positions_sample[:, 0]
        axes[2, 0].plot(time_steps, delta_diff, color='tab:red', linewidth=2)
        axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[2, 0].set_xlabel("Time Step", fontsize=11)
        axes[2, 0].set_ylabel("Delta Difference", fontsize=11)
        axes[2, 0].set_title(f"RL Delta - Practitioner Delta (Path {path_idx})", fontsize=12)
        axes[2, 0].grid(True, alpha=0.3)
        
        # PLOT 6: Terminal Error Distribution
        axes[2, 1].hist(terminal_hedge_error_rl, bins=50, color="tab:blue", alpha=0.7,
                        edgecolor='black', label='RL')
        axes[2, 1].hist(terminal_hedge_error_hn, bins=50, color="tab:orange", alpha=0.7,
                        edgecolor='black', label='Practitioner')
        axes[2, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[2, 1].set_xlabel("Terminal Hedge Error", fontsize=11)
        axes[2, 1].set_ylabel("Frequency", fontsize=11)
        
        greek_labels = {1: 'Delta', 2: 'Delta-Gamma', 3: 'Delta-Gamma-Vega', 4: 'Delta-Gamma-Vega-Theta'}
        hedged_option_type = config["hedged_option"]["type"].capitalize()
        
        title_text = (f"Episode {episode} - {n_inst} Instruments ({greek_labels[n_inst]}) - Hedging {hedged_option_type}\n"
                    f"RL: MSE={rl_metrics['mse']:.4f} | SMSE={rl_metrics['smse']:.6f} | CVaR95={rl_metrics['cvar_95']:.4f}\n"
                    f"Prac: MSE={mse_hn:.4f} | SMSE={smse_hn:.6f} | CVaR95={cvar_95_hn:.4f}")
        axes[2, 1].set_title(title_text, fontsize=10)
        axes[2, 1].legend(fontsize=10)
        axes[2, 1].grid(True, alpha=0.3)
        
        # Save plot
        fig.tight_layout()
        save_path = config["output"]["plot_save_path"].format(
            n_inst=n_inst,
            episode=episode
        )
        plt.savefig(save_path, dpi=config["output"]["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Plotting failed: {type(e).__name__}: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
