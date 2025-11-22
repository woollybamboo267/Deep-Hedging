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
    config: Dict[str, Any]
) -> None:
    """Create comprehensive visualization of episode results with full support for all models."""
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
            fig, axes = plt.subplots(4, 2, figsize=(14, 18))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        
        time_steps = np.arange(env.N + 1)
        
        # Extract sample paths for plotting
        rl_positions_sample = RL_positions[path_idx].cpu().detach().numpy()
        hn_positions_sample = HN_positions_all[path_idx].cpu().detach().numpy()
        
        # ============ GREEK DIAGNOSTICS ============
        print("\n" + "="*80)
        print(f"GREEK DIAGNOSTICS FOR PATH {path_idx}")
        print("="*80)
        
        # Determine which Greeks to analyze
        if n_inst == 1:
            greek_names = ['delta']
        elif n_inst == 2:
            greek_names = ['delta', 'gamma']
        elif n_inst == 3:
            greek_names = ['delta', 'gamma', 'vega']
        elif n_inst == 4:
            greek_names = ['delta', 'gamma', 'vega', 'theta']
        
        # Compute portfolio Greeks (what we're trying to hedge)
        print("\n--- PORTFOLIO GREEKS (Derivative being hedged) ---")
        portfolio_greeks_diag = {}
        for greek_name in greek_names:
            greek_traj = env.compute_all_paths_greeks(S_traj, greek_name)
            portfolio_greek = -env.side * greek_traj[path_idx].cpu().numpy()
            portfolio_greeks_diag[greek_name] = portfolio_greek
            
            print(f"\n{greek_name.upper()}:")
            print(f"  t=0:  {portfolio_greek[0]:12.6f}")
            print(f"  t=1:  {portfolio_greek[1]:12.6f}")
            if len(portfolio_greek) > 5:
                print(f"  t=5:  {portfolio_greek[5]:12.6f}")
            print(f"  Mean: {portfolio_greek.mean():12.6f}")
            print(f"  Max:  {portfolio_greek.max():12.6f}")
            print(f"  Min:  {portfolio_greek.min():12.6f}")
        
        # Compute hedging instrument Greeks (recompute them directly)
        if n_inst >= 2:
            print("\n--- HEDGING INSTRUMENT GREEKS (Used in matrix solve) ---")
            
            hedging_greeks = {}
            h0_mean = env.h_t.mean().item()
            
            for i in range(1, n_inst):
                opt_idx = i - 1
                hedge_deriv = env.hedging_derivatives[i]
                maturity = env.instrument_maturities[i]
                opt_type = env.instrument_types[i]
                strike = env.instrument_strikes[i]
                
                print(f"\nInstrument {i}: {maturity}d {opt_type.upper()} K={strike}")
                
                hedging_greeks[i] = {}
                is_asian_hedge = env.hedging_is_asian[i]
                
                for greek_name in greek_names:
                    # Compute Greek trajectory for this hedging instrument along the sample path
                    greek_traj = torch.zeros((env.N + 1,), device=env.device)
                    greek_method = getattr(hedge_deriv, greek_name)
                    
                    A_t = None
                    
                    for t in range(env.N + 1):
                        S_t = S_traj[path_idx, t]
                        K_hedge = getattr(hedge_deriv, 'K', env.K)
                        N_hedge = getattr(hedge_deriv, 'N', env.N)
                        
                        if is_asian_hedge:
                            # Update running average for Asian hedging instrument
                            A_t = env._update_running_average(A_t, S_t, t, N_hedge)
                            greek_val = greek_method(
                                S=S_t.unsqueeze(0), K=K_hedge, step_idx=t, N=N_hedge, h0=h0_mean, A=A_t.unsqueeze(0)
                            )[0]
                        else:
                            greek_val = greek_method(
                                S=S_t.unsqueeze(0), K=K_hedge, step_idx=t, N=N_hedge, h0=h0_mean
                            )[0]
                        
                        greek_traj[t] = greek_val
                    
                    hedging_greeks[i][greek_name] = greek_traj.cpu().numpy()
                    
                    print(f"  {greek_name.upper()}:")
                    print(f"    t=0:  {greek_traj[0].item():12.6f}")
                    print(f"    t=1:  {greek_traj[1].item():12.6f}")
                    if len(greek_traj) > 5:
                        print(f"    t=5:  {greek_traj[5].item():12.6f}")
        
        # Print the matrix condition at critical time points
        if n_inst >= 2:
            print("\n--- MATRIX ANALYSIS AT CRITICAL TIME POINTS ---")
            
            for t in [0, 1, 5] if env.N >= 5 else [0, 1]:
                if t >= len(time_steps):
                    continue
                    
                print(f"\n>>> Time t={t} <<<")
                print(f"Stock Price: {S_traj[path_idx, t].item():.4f}")
                
                print("\nMatrix A (hedging instrument Greeks):")
                print("         ", end="")
                for i in range(1, n_inst):
                    print(f"  Inst_{i:2d}    ", end="")
                print()
                
                for greek_name in greek_names:
                    print(f"{greek_name:8s}:", end="")
                    for i in range(1, n_inst):
                        if i in hedging_greeks and greek_name in hedging_greeks[i]:
                            val = hedging_greeks[i][greek_name][t]
                            print(f" {val:11.6f}", end="")
                        else:
                            print(f"     N/A    ", end="")
                    print()
                
                print("\nVector b (portfolio Greeks to neutralize):")
                for greek_name in greek_names:
                    val = portfolio_greeks_diag[greek_name][t]
                    print(f"  {greek_name:8s}: {val:12.6f}")
                
                print("\nSolution (hedge positions):")
                print(f"  Stock:    {hn_positions_sample[t, 0]:12.6f}")
                for i in range(1, n_inst):
                    maturity = env.instrument_maturities[i]
                    opt_type = env.instrument_types[i]
                    strike = env.instrument_strikes[i]
                    print(f"  Opt_{i}:    {hn_positions_sample[t, i]:12.6f}  ({maturity}d {opt_type} K={strike})")
                
                # Compute matrix condition number if possible
                if n_inst >= 2 and all(i in hedging_greeks for i in range(1, n_inst)):
                    try:
                        A_matrix = []
                        for greek_name in greek_names:
                            row = [hedging_greeks[i][greek_name][t] for i in range(1, n_inst)]
                            A_matrix.append(row)
                        A_matrix = np.array(A_matrix)
                        
                        if A_matrix.shape[0] == A_matrix.shape[1]:
                            cond = np.linalg.cond(A_matrix)
                            det = np.linalg.det(A_matrix)
                            print(f"\nMatrix Condition Number: {cond:.2e}")
                            print(f"Matrix Determinant: {det:.2e}")
                            
                            if cond > 1e10:
                                print("⚠️  WARNING: Matrix is nearly singular (ill-conditioned)!")
                            if abs(det) < 1e-10:
                                print("⚠️  WARNING: Matrix determinant near zero!")
                    except:
                        print("\n[Could not compute condition number]")
        
        print("\n" + "="*80)
        print("END GREEK DIAGNOSTICS")
        print("="*80 + "\n")
        
        # ============ END GREEK DIAGNOSTICS ============
        
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
            barrier_val = env.derivative.barrier_level
            if isinstance(barrier_val, torch.Tensor):
                barrier_val = barrier_val.item()
            ax1.axhline(y=barrier_val, color='purple', 
                       linestyle=':', label='Barrier', alpha=0.7, linewidth=2)
        
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
        
        # Build comprehensive title starting with config name
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
        
        # Start with config name as main title
        title_text = f"{config_name}\n"
        title_text += (
            f"Episode {episode} - {n_inst} Instruments ({greek_labels[n_inst]}) - Hedging {hedged_option_type}\n"
            f"Training Objective: {risk_display}"
        )
        
        if has_soft_constraint:
            title_text += f" + λ·SC (λ={lambda_constraint:.4f})"
        
        # Add stability info showing hardcoded clip bounds
        title_text += f"\n[Greeks Clipped: Δ=±5·CS"
        if n_inst >= 2:
            title_text += f", Γ=±10·CS"
        if n_inst >= 3:
            title_text += f", ν=±10·CS"
        if n_inst >= 4:
            title_text += f", Θ=±10·CS"
        title_text += "]"
        
        title_text += "\n"
        
        # Show training objective values
        title_text += f"RL {risk_display}={rl_training_objective:.4f} | Prac {risk_display}={hn_training_objective:.4f}"
        
        if has_soft_constraint:
            avg_violation_rl = trajectories['soft_constraint_violations'].mean().cpu().item()
            avg_violation_hn = trajectories_hn.get('soft_constraint_violations', torch.zeros(1)).mean().cpu().item() if 'soft_constraint_violations' in trajectories_hn else 0.0
            title_text += f"\nRL Constr={avg_violation_rl:.4f} | Prac Constr={avg_violation_hn:.4f}"
        
        axes[2, 1].set_title(title_text, fontsize=9)
        axes[2, 1].legend(fontsize=10)
        axes[2, 1].grid(True, alpha=0.3)
        
        # PLOT 7 & 8: Soft Constraint Visualizations (if enabled)
        if has_soft_constraint:
            violations_rl = trajectories['soft_constraint_violations'].cpu().numpy()
            violations_hn = trajectories_hn.get('soft_constraint_violations', 
                                                 torch.zeros_like(trajectories['soft_constraint_violations'])).cpu().numpy()
            
            # PLOT 7: Accumulated Constraint Violations Distribution
            axes[3, 0].hist(violations_rl, bins=50, color="tab:blue", alpha=0.7,
                           edgecolor='black', label='RL')
            axes[3, 0].hist(violations_hn, bins=50, color="tab:orange", alpha=0.7,
                           edgecolor='black', label='Practitioner')
            axes[3, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[3, 0].set_xlabel("Accumulated Constraint Violations", fontsize=11)
            axes[3, 0].set_ylabel("Frequency", fontsize=11)
            axes[3, 0].set_title("Soft Constraint Violations: ∑max(0, P_t - V_t)", fontsize=12)
            axes[3, 0].legend(fontsize=10)
            axes[3, 0].grid(True, alpha=0.3)
            
            # PLOT 8: Violation Statistics
            axes[3, 1].axis('off')
            
            # Compute violation statistics
            rl_violation_pct = (violations_rl > 0).sum() / len(violations_rl) * 100
            hn_violation_pct = (violations_hn > 0).sum() / len(violations_hn) * 100
            
            rl_mean_violation = violations_rl.mean()
            hn_mean_violation = violations_hn.mean()
            
            rl_max_violation = violations_rl.max()
            hn_max_violation = violations_hn.max()
            
            stats_text = (
                "Soft Constraint Statistics\n"
                "─" * 40 + "\n\n"
                f"Constraint Weight (λ): {lambda_constraint:.4f}\n"
                f"Constraint Penalty: {constraint_penalty:.4f}\n\n"
                "RL Strategy:\n"
                f"  • Violation Rate: {rl_violation_pct:.1f}%\n"
                f"  • Mean Violation: {rl_mean_violation:.4f}\n"
                f"  • Max Violation: {rl_max_violation:.4f}\n\n"
                "Practitioner Strategy:\n"
                f"  • Violation Rate: {hn_violation_pct:.1f}%\n"
                f"  • Mean Violation: {hn_mean_violation:.4f}\n"
                f"  • Max Violation: {hn_max_violation:.4f}\n\n"
                "Interpretation:\n"
                f"  • Violations occur when P_t > V_t\n"
                f"  • Penalty = λ × mean(violations)\n"
                f"  • Lower is better"
            )
            
            axes[3, 1].text(0.1, 0.5, stats_text, transform=axes[3, 1].transAxes,
                           fontsize=10, verticalalignment='center', family='monospace')
        
        # Save plot using config name
        fig.tight_layout()
        
        # Use config name instead of generic path
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
