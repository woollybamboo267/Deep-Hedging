import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import os
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def compute_practitioner_benchmark(
    env: Any,
    S_traj: torch.Tensor,
    O_traj: Dict[int, torch.Tensor],
    n_hedging_instruments: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], np.ndarray]:
    """
    Compute the practitioner benchmark hedge with stability controls.
    
    Greeks are automatically clipped to prevent numerical explosions:
    - Delta: ±5 × contract_size
    - Gamma: ±10 × contract_size
    - Vega: ±10 × contract_size
    - Theta: ±10 × contract_size
    
    Args:
        env: Trading environment
        S_traj: Stock price trajectories
        O_traj: Option price trajectories
        n_hedging_instruments: Number of hedging instruments
    
    Returns:
        Tuple of (positions, trajectories, terminal_hedge_errors)
    """
    S_np = S_traj.cpu().numpy()
    
    # Hardcoded clipping bounds (multiples of contract_size)
    DELTA_CLIP = 1000
    GAMMA_CLIP = 1000
    VEGA_CLIP = 1000
    THETA_CLIP = 1000
    
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
    
    # Define clipping bounds for each Greek
    clip_bounds = {
        'delta': DELTA_CLIP * env.contract_size,
        'gamma': GAMMA_CLIP * env.contract_size,
        'vega': VEGA_CLIP * env.contract_size,
        'theta': THETA_CLIP * env.contract_size
    }
    
    # Compute portfolio Greeks with stability controls
    portfolio_greeks = {}
    instability_detected = False
    
    for greek_name in greek_names:
        greek_traj = env.compute_all_paths_greeks(S_traj, greek_name)
        portfolio_greeks[greek_name] = -env.side * greek_traj
        
        # OPTION 2: Detect instabilities
        # Check for NaN or Inf
        has_nan = torch.isnan(portfolio_greeks[greek_name]).any()
        has_inf = torch.isinf(portfolio_greeks[greek_name]).any()
        
        if has_nan or has_inf:
            nan_count = torch.isnan(portfolio_greeks[greek_name]).sum().item()
            inf_count = torch.isinf(portfolio_greeks[greek_name]).sum().item()
            total_elements = portfolio_greeks[greek_name].numel()
            
            logger.warning(
                f"[Practitioner Benchmark] {greek_name.upper()} contains "
                f"{nan_count} NaN ({nan_count/total_elements*100:.2f}%) and "
                f"{inf_count} Inf ({inf_count/total_elements*100:.2f}%) values"
            )
            
            # Replace NaN/Inf with zeros
            portfolio_greeks[greek_name] = torch.nan_to_num(
                portfolio_greeks[greek_name], 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
            )
            instability_detected = True
        
        # Check for extreme values before clipping
        threshold = clip_bounds[greek_name]
        extreme_mask = torch.abs(portfolio_greeks[greek_name]) > threshold
        
        if extreme_mask.any():
            n_extreme = extreme_mask.sum().item()
            pct_extreme = n_extreme / portfolio_greeks[greek_name].numel() * 100
            max_val = torch.abs(portfolio_greeks[greek_name]).max().item()
            
            logger.warning(
                f"[Practitioner Benchmark] {greek_name.upper()} has {n_extreme} "
                f"extreme values ({pct_extreme:.2f}% > {threshold:.2f}), "
                f"max absolute value: {max_val:.2f}"
            )
            instability_detected = True
        
        # OPTION 1: Clip Greeks to reasonable bounds
        original_mean = portfolio_greeks[greek_name].abs().mean().item()
        
        portfolio_greeks[greek_name] = torch.clamp(
            portfolio_greeks[greek_name], 
            -clip_bounds[greek_name], 
            clip_bounds[greek_name]
        )
        
        clipped_mean = portfolio_greeks[greek_name].abs().mean().item()
        
        # Log clipping impact
        if abs(original_mean - clipped_mean) / (original_mean + 1e-8) > 0.01:
            logger.info(
                f"[Practitioner Benchmark] {greek_name.upper()} clipped: "
                f"mean |value| changed from {original_mean:.4f} to {clipped_mean:.4f} "
                f"(bounds: ±{clip_bounds[greek_name]:.2f})"
            )
    
    if instability_detected:
        logger.warning(
            "[Practitioner Benchmark] Numerical instabilities detected and corrected. "
            "This is common near expiry or at barriers for exotic derivatives."
        )
    
    # Solve for optimal hedge positions
    HN_positions_all = env.compute_hn_option_positions(S_traj, portfolio_greeks)
    
    # Additional stability check on positions
    if torch.isnan(HN_positions_all).any() or torch.isinf(HN_positions_all).any():
        logger.error(
            "[Practitioner Benchmark] Hedge positions contain NaN/Inf after solving. "
            "Replacing with zeros."
        )
        HN_positions_all = torch.nan_to_num(HN_positions_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Simulate hedge strategy
    _, trajectories_hn = env.simulate_full_trajectory(HN_positions_all, O_traj)
    
    # Compute terminal hedge errors
    S_final = trajectories_hn['S'][:, -1]
    
    # Compute payoff based on derivative type
    if hasattr(env.derivative, 'option_type'):
        opt_type = env.derivative.option_type.lower()
    else:
        opt_type = env.option_type.lower()
    
    # Handle Asian option payoff
    if hasattr(env, 'is_asian_hedged') and env.is_asian_hedged:
        # For Asian options, need running average A
        # Reconstruct it from trajectory
        A_hedged = env._reconstruct_running_average(trajectories_hn['S'])
        if opt_type == "call":
            payoff = torch.clamp(A_hedged - env.K, min=0.0)
        else:
            payoff = torch.clamp(env.K - A_hedged, min=0.0)
    else:
        # Standard European payoff
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


def clip_outliers_to_last_valid(positions: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """
    Clip extreme outliers in position trajectories by replacing them with the last valid value.
    
    Args:
        positions: Array of shape (n_timesteps, n_instruments)
        n_std: Number of standard deviations beyond which to clip (default 3.0)
    
    Returns:
        Clipped positions array
    """
    clipped = positions.copy()
    n_timesteps, n_instruments = positions.shape
    
    for inst_idx in range(n_instruments):
        pos = positions[:, inst_idx]
        
        # Compute statistics
        mean = np.mean(pos)
        std = np.std(pos)
        
        if std < 1e-10:  # Avoid division by zero
            continue
        
        # Identify outliers
        z_scores = np.abs((pos - mean) / std)
        outlier_mask = z_scores > n_std
        
        if not outlier_mask.any():
            continue
        
        # Replace outliers with last valid value
        last_valid = pos[0]
        for t in range(n_timesteps):
            if outlier_mask[t]:
                clipped[t, inst_idx] = last_valid
                logger.info(
                    f"Clipped outlier at t={t}, instrument {inst_idx}: "
                    f"{pos[t]:.4f} -> {last_valid:.4f} (z-score: {z_scores[t]:.2f})"
                )
            else:
                last_valid = pos[t]
    
    return clipped


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
    
    # Handle Asian option payoff
    if hasattr(env, 'is_asian_hedged') and env.is_asian_hedged:
        # For Asian options, need running average A
        A_hedged = env._reconstruct_running_average(trajectories['S'])
        if opt_type == "call":
            payoff = torch.clamp(A_hedged - env.K, min=0.0)
        else:
            payoff = torch.clamp(env.K - A_hedged, min=0.0)
    else:
        # Standard European payoff
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
    mae_rl = float(np.mean(np.abs(terminal_hedge_error_rl)))
    
    metrics = {
        'mse': mse_rl,
        'smse': smse_rl,
        'cvar_95': cvar_95_rl,
        'mae': mae_rl
    }
    
    return terminal_hedge_error_rl, metrics


def compute_risk_measure_value(terminal_errors: np.ndarray, risk_measure: str, alpha: Optional[float] = None) -> float:
    """
    Compute the value of a specific risk measure on terminal errors.
    
    Args:
        terminal_errors: Array of terminal hedging errors
        risk_measure: 'mse', 'smse', 'cvar', 'var', or 'mae'
        alpha: Confidence level for CVaR or VaR (e.g., 0.95)
    
    Returns:
        Risk measure value
    """
    if risk_measure == 'mse':
        return float(np.mean(terminal_errors ** 2))
    
    elif risk_measure == 'smse':
        positive_mask = (terminal_errors >= 0)
        return float(np.mean((terminal_errors ** 2) * positive_mask))
    
    elif risk_measure == 'cvar':
        if alpha is None:
            alpha = 0.95
        sorted_errors = np.sort(terminal_errors)[::-1]  # Descending
        n_tail = max(1, int(np.ceil(len(sorted_errors) * (1 - alpha))))
        return float(np.mean(sorted_errors[:n_tail]))
    
    elif risk_measure == 'var':
        if alpha is None:
            alpha = 0.95
        return float(np.quantile(terminal_errors, alpha))
    
    elif risk_measure == 'mae':
        return float(np.mean(np.abs(terminal_errors)))
    
    else:
        # Default to MSE
        return float(np.mean(terminal_errors ** 2))


def plot_episode_results(
    episode: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    use_clipped_practitioner_errors: bool = True
) -> None:
    """
    Create comprehensive visualization of episode results with full support for all models.
    
    Args:
        episode: Episode number
        metrics: Dictionary containing trajectories and positions
        config: Configuration dictionary
        use_clipped_practitioner_errors: If True, recalculate practitioner errors using clipped positions
    """
    try:
        logger.info(f"Generating plots for episode {episode}")
        
        # Extract data from metrics
        env = metrics['env']
        RL_positions = metrics['RL_positions']
        trajectories = metrics['trajectories']
        S_traj = metrics['S_traj']
        V_traj = metrics['V_traj']
        O_traj = metrics['O_traj']
        
        # Extract training configuration
        risk_measure = metrics.get('risk_measure', 'mse')
        lambda_constraint = metrics.get('lambda_constraint', 0.0)
        risk_loss = metrics.get('risk_loss', 0.0)
        constraint_penalty = metrics.get('constraint_penalty', 0.0)
        total_loss = metrics.get('loss', 0.0)
        
        # Get risk measure alpha if applicable
        risk_config = config.get("risk_measure", {})
        alpha = risk_config.get("alpha", 0.95)
        
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
        
        # Compute practitioner benchmark with stability controls
        logger.info("Computing practitioner benchmark with stability controls...")
        HN_positions_all, trajectories_hn, terminal_hedge_error_hn = \
            compute_practitioner_benchmark(env, S_traj, O_traj, n_inst)
        
        # Apply outlier clipping to ALL practitioner positions if enabled
        if use_clipped_practitioner_errors:
            logger.info("Applying outlier clipping to practitioner positions (z > 3.0 std)...")
            HN_positions_all_np = HN_positions_all.cpu().detach().numpy()
            HN_positions_clipped = np.zeros_like(HN_positions_all_np)
            
            n_paths = HN_positions_all_np.shape[0]
            total_clipped = 0
            
            for path_idx in range(n_paths):
                clipped_path = clip_outliers_to_last_valid(HN_positions_all_np[path_idx], n_std=3.0)
                HN_positions_clipped[path_idx] = clipped_path
                
                # Count how many positions were clipped
                total_clipped += np.sum(np.abs(clipped_path - HN_positions_all_np[path_idx]) > 1e-10)
            
            logger.info(f"Clipped {total_clipped} outlier positions across {n_paths} paths")
            
            # Convert back to tensor
            HN_positions_all = torch.from_numpy(HN_positions_clipped).to(HN_positions_all.device).float()
            
            # Recalculate trajectories and terminal errors with clipped positions
            _, trajectories_hn = env.simulate_full_trajectory(HN_positions_all, O_traj)
            
            # Recompute terminal hedge errors
            S_final = trajectories_hn['S'][:, -1]
            
            if hasattr(env.derivative, 'option_type'):
                opt_type = env.derivative.option_type.lower()
            else:
                opt_type = env.option_type.lower()
            
            if hasattr(env, 'is_asian_hedged') and env.is_asian_hedged:
                A_hedged = env._reconstruct_running_average(trajectories_hn['S'])
                if opt_type == "call":
                    payoff = torch.clamp(A_hedged - env.K, min=0.0)
                else:
                    payoff = torch.clamp(env.K - A_hedged, min=0.0)
            else:
                if opt_type == "call":
                    payoff = torch.clamp(S_final - env.K, min=0.0)
                else:
                    payoff = torch.clamp(env.K - S_final, min=0.0)
            
            payoff = payoff * env.contract_size
            
            terminal_value_hn = trajectories_hn['B'][:, -1] + HN_positions_all[:, -1, 0] * S_final
            for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
                terminal_value_hn += HN_positions_all[:, -1, i] * O_traj[i - 1][:, -1]
            
            terminal_hedge_error_hn = (terminal_value_hn - env.side * payoff).cpu().detach().numpy()
            
            logger.info("Practitioner errors recalculated with clipped positions")
        
        # Compute HN metrics (standard)
        mse_hn = float(np.mean(terminal_hedge_error_hn ** 2))
        smse_hn = mse_hn / (env.S0 ** 2)
        cvar_95_hn = float(np.mean(np.sort(terminal_hedge_error_hn ** 2)[-int(0.05 * env.M):]))
        mae_hn = float(np.mean(np.abs(terminal_hedge_error_hn)))
        
        # Compute the ACTUAL training objective for both strategies
        rl_training_objective = compute_risk_measure_value(terminal_hedge_error_rl, risk_measure, alpha)
        hn_training_objective = compute_risk_measure_value(terminal_hedge_error_hn, risk_measure, alpha)
        
        # Determine if soft constraints are active
        has_soft_constraint = lambda_constraint > 0 and 'soft_constraint_violations' in trajectories
        
        # Create figure with appropriate number of subplots
        if has_soft_constraint:
            fig = plt.figure(figsize=(16, 18))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 14))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        time_steps = np.arange(env.N + 1)
        
        # Extract sample paths for plotting (apply clipping for visualization only)
        rl_positions_sample = RL_positions[path_idx].cpu().detach().numpy()
        hn_positions_sample = HN_positions_all[path_idx].cpu().detach().numpy()
        
        # Note: If use_clipped_practitioner_errors=True, HN_positions_all is already clipped
        # If False, we still clip just for visualization
        if not use_clipped_practitioner_errors:
            hn_positions_sample = clip_outliers_to_last_valid(hn_positions_sample, n_std=3.0)
        
        # PLOT 1: Stock Delta Comparison (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time_steps, rl_positions_sample[:, 0], label='RL Delta',
                 linewidth=2, color='tab:blue')
        ax1.plot(time_steps, hn_positions_sample[:, 0], label='Practitioner Delta',
                 linewidth=2, linestyle='--', alpha=0.8, color='tab:orange')
        ax1.set_xlabel("Time Step", fontsize=11)
        ax1.set_ylabel("Delta", fontsize=11)
        ax1.set_title(f"Stock Delta: Practitioner vs RL (Path {path_idx})", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # PLOT 2: Option Positions Comparison (right column, row 1)
        ax2 = fig.add_subplot(gs[0, 2])
        if n_inst >= 2:
            for i in range(1, n_inst):
                maturity = env.instrument_maturities[i]
                opt_type = env.instrument_types[i]
                strike = env.instrument_strikes[i]
                label_suffix = f'{maturity}d {opt_type.upper()} K={strike}'
                
                ax2.plot(time_steps, rl_positions_sample[:, i],
                        label=f'RL {label_suffix}', linewidth=2)
                ax2.plot(time_steps, hn_positions_sample[:, i],
                        label=f'Prac {label_suffix}', linewidth=2,
                        linestyle='--', alpha=0.8)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel("Time Step", fontsize=11)
            ax2.set_ylabel("Option Contracts", fontsize=11)
            ax2.set_title(f"Option Positions (Path {path_idx})", fontsize=12)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No option positions\n(Delta hedge only)',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Option Positions", fontsize=12)
        
        # PLOT 3: Stock Price Trajectory (left, row 2)
        ax3 = fig.add_subplot(gs[1, 0])
        color_stock = 'tab:green'
        ax3.plot(time_steps, S_traj[path_idx].cpu().detach().numpy(),
                label='Stock Price', color=color_stock, linewidth=2)
        ax3.axhline(y=env.K, color='r', linestyle='--', label='Strike', alpha=0.7)
        
        if hasattr(env.derivative, 'barrier_level'):
            barrier_val = env.derivative.barrier_level
            if isinstance(barrier_val, torch.Tensor):
                barrier_val = barrier_val.item()
            ax3.axhline(y=barrier_val, color='purple', 
                       linestyle=':', label='Barrier', alpha=0.7, linewidth=2)
        
        ax3.set_xlabel("Time Step", fontsize=11)
        ax3.set_ylabel("Stock Price", fontsize=11)
        ax3.set_title(f"Stock Price (Path {path_idx})", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # PLOT 4: Derivative Price (middle, row 2)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_steps, V_traj[path_idx].cpu().detach().numpy(),
                label=f'{derivative_type} Price', color='tab:blue', linewidth=2)
        ax4.set_xlabel("Time Step", fontsize=11)
        ax4.set_ylabel(f"{derivative_type} Price", fontsize=11)
        ax4.set_title(f"{derivative_type} Price (Path {path_idx})", fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # PLOT 5: Hedging Instrument Prices (right, row 2)
        ax5 = fig.add_subplot(gs[1, 2])
        if n_inst >= 2:
            for i in range(1, n_inst):
                opt_idx = i - 1
                if opt_idx not in O_traj:
                    continue
                    
                maturity = env.instrument_maturities[i]
                opt_type = env.instrument_types[i]
                strike = env.instrument_strikes[i]
                
                option_prices = O_traj[opt_idx][path_idx].cpu().detach().numpy()
                ax5.plot(time_steps, option_prices,
                        label=f'{maturity}d {opt_type.upper()} K={strike}', linewidth=2)
        
        ax5.set_xlabel("Time Step", fontsize=11)
        ax5.set_ylabel("Option Price", fontsize=11)
        ax5.set_title(f"Hedging Instrument Prices (Path {path_idx})", fontsize=12)
        if n_inst >= 2:
            ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # PLOT 6: Position Difference (left, row 3)
        ax6 = fig.add_subplot(gs[2, 0])
        delta_diff = rl_positions_sample[:, 0] - hn_positions_sample[:, 0]
        ax6.plot(time_steps, delta_diff, color='tab:red', linewidth=2)
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax6.set_xlabel("Time Step", fontsize=11)
        ax6.set_ylabel("Delta Difference", fontsize=11)
        ax6.set_title(f"RL - Practitioner Delta (Path {path_idx})", fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # PLOT 7: Terminal Error Distribution (middle+right, row 3, spans 2 columns)
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.hist(terminal_hedge_error_rl, bins=50, color="tab:blue", alpha=0.7,
                edgecolor='black', label='RL')
        ax7.hist(terminal_hedge_error_hn, bins=50, color="tab:orange", alpha=0.7,
                edgecolor='black', label='Practitioner')
        ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax7.set_xlabel("Terminal Hedge Error", fontsize=11)
        ax7.set_ylabel("Frequency", fontsize=11)
        
        # Build title
        config_name = config.get("config_name", "model")
        greek_labels = {1: 'Delta', 2: 'Delta-Gamma', 3: 'Delta-Gamma-Vega', 4: 'Delta-Gamma-Vega-Theta'}
        hedged_option_type = config["hedged_option"]["type"].capitalize()
        
        risk_display_names = {
            'mse': 'MSE',
            'smse': 'SMSE',
            'cvar': f'CVaR_{int(alpha*100)}',
            'var': f'VaR_{int(alpha*100)}',
            'mae': 'MAE'
        }
        
        risk_display = risk_display_names.get(risk_measure, risk_measure.upper())
        
        title_text = f"{config_name} | Episode {episode}\n"
        title_text += (
            f"{n_inst} Instruments ({greek_labels[n_inst]}) - Hedging {hedged_option_type}\n"
            f"Training Objective: {risk_display}"
        )
        
        if has_soft_constraint:
            title_text += f" + λ·SC (λ={lambda_constraint:.4f})"
        
        if use_clipped_practitioner_errors:
            title_text += " [Prac: Clipped Positions]"
        
        title_text += "\n"
        title_text += f"RL {risk_display}={rl_training_objective:.4f} | Prac {risk_display}={hn_training_objective:.4f}"
        
        ax7.set_title(title_text, fontsize=10)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # PLOT 8 & 9: Soft Constraint Visualizations (if enabled)
        if has_soft_constraint:
            violations_rl = trajectories['soft_constraint_violations'].cpu().numpy()
            violations_hn = trajectories_hn.get('soft_constraint_violations', 
                                                 torch.zeros_like(trajectories['soft_constraint_violations'])).cpu().numpy()
            
            # PLOT 8: Constraint Violations Distribution (left, row 4)
            ax8 = fig.add_subplot(gs[3, :2])
            ax8.hist(violations_rl, bins=50, color="tab:blue", alpha=0.7,
                    edgecolor='black', label='RL')
            ax8.hist(violations_hn, bins=50, color="tab:orange", alpha=0.7,
                    edgecolor='black', label='Practitioner')
            ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax8.set_xlabel("Accumulated Constraint Violations", fontsize=11)
            ax8.set_ylabel("Frequency", fontsize=11)
            ax8.set_title("Soft Constraint Violations: ∑max(0, P_t - V_t)", fontsize=12)
            ax8.legend(fontsize=10)
            ax8.grid(True, alpha=0.3)
            
            # PLOT 9: Statistics Table (right, row 4)
            ax9 = fig.add_subplot(gs[3, 2])
            ax9.axis('off')
            
            # Compute statistics
            rl_violation_pct = (violations_rl > 0).sum() / len(violations_rl) * 100
            hn_violation_pct = (violations_hn > 0).sum() / len(violations_hn) * 100
            
            rl_mean_violation = violations_rl.mean()
            hn_mean_violation = violations_hn.mean()
            
            rl_max_violation = violations_rl.max()
            hn_max_violation = violations_hn.max()
            
            # Create table
            table_data = [
                ['Metric', 'RL', 'Practitioner'],
                ['Violation Rate', f'{rl_violation_pct:.1f}%', f'{hn_violation_pct:.1f}%'],
                ['Mean Violation', f'{rl_mean_violation:.4f}', f'{hn_mean_violation:.4f}'],
                ['Max Violation', f'{rl_max_violation:.4f}', f'{hn_max_violation:.4f}'],
                ['Penalty Weight (λ)', f'{lambda_constraint:.4f}', '—'],
                ['Constraint Penalty', f'{constraint_penalty:.4f}', '—']
            ]
            
            table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.4, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header row
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data)):
                for j in range(3):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            ax9.set_title("Soft Constraint Statistics", fontsize=12, weight='bold', pad=20)
        
        # Save plot using config name
        config_name = config.get("config_name", "model")
        output_dir = os.path.dirname(config["output"]["plot_save_path"])
        save_path = os.path.join(output_dir, f"{config_name}_episode_{episode}.png")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=config["output"]["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Plotting failed: {type(e).__name__}: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
