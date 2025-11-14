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
    """
    Compute the practitioner benchmark hedge with extensive diagnostics.
    """
    logger.info("=" * 80)
    logger.info(f"COMPUTING ANALYTICAL HEDGE for {env.M} paths using {type(env.derivative).__name__}")
    logger.info("=" * 80)
    
    # ========== DIAGNOSTIC: BARRIER BREACH ANALYSIS ==========
    logger.info("DIAGNOSTIC: Analyzing barrier breach patterns")
    S_np = S_traj.cpu().numpy()
    barrier_level = env.derivative.barrier_level
    
    breached_paths = []
    breach_times = []
    for path_idx in range(env.M):
        breach_idx = np.where(S_np[path_idx, :] >= barrier_level)[0]
        if len(breach_idx) > 0:
            breached_paths.append(path_idx)
            breach_times.append(breach_idx[0])
    
    n_breached = len(breached_paths)
    breach_pct = 100 * n_breached / env.M
    
    logger.info(f"  → Paths that breached barrier: {n_breached}/{env.M} ({breach_pct:.1f}%)")
    logger.info(f"  → Paths that never breached: {env.M - n_breached}/{env.M} ({100-breach_pct:.1f}%)")
    
    if n_breached > 0:
        logger.info(f"  → Average breach time: {np.mean(breach_times):.1f} steps")
        logger.info(f"  → Earliest breach: step {np.min(breach_times)}")
        logger.info(f"  → Latest breach: step {np.max(breach_times)}")
    
    breach_mask = np.array([i in breached_paths for i in range(env.M)])

    # ========== STEP 1: Determine Greeks to Hedge ==========
    logger.info(f"STEP 1/5: Determining Greeks to hedge based on {n_hedging_instruments} instruments")
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
    
    logger.info(f"  → Greeks to hedge: {greek_names}")

    # ========== STEP 2: Compute Portfolio Greeks with Diagnostics ==========
    logger.info(f"STEP 2/5: Computing portfolio Greeks along all price paths")
    portfolio_greeks = {}
    greek_stats = {}
    start_time = time.time()

    with tqdm(total=len(greek_names), desc="Computing Greeks", ncols=100) as pbar:
        for idx, greek_name in enumerate(greek_names, 1):
            iter_start = time.time()
            logger.info(f"  → Computing {greek_name} ({idx}/{len(greek_names)})...")
            
            greek_traj = env.compute_all_paths_greeks(S_traj, greek_name)
            portfolio_greeks[greek_name] = -env.side * greek_traj

            # DIAGNOSTIC: Compute Greek statistics
            greek_np = greek_traj.cpu().numpy()
            greek_stats[greek_name] = {
                'mean': greek_np.mean(),
                'std': greek_np.std(),
                'min': greek_np.min(),
                'max': greek_np.max(),
                'nan_count': np.isnan(greek_np).sum(),
                'inf_count': np.isinf(greek_np).sum(),
                'zero_count': (greek_np == 0).sum(),
                'total_count': greek_np.size
            }

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            remaining = (len(greek_names) - idx) * iter_time
            
            logger.info(f"    ✓ {greek_name} computed: shape {greek_traj.shape}")
            logger.info(f"    ✓ Range: [{greek_stats[greek_name]['min']:.6f}, {greek_stats[greek_name]['max']:.6f}]")
            logger.info(f"    ✓ Mean: {greek_stats[greek_name]['mean']:.6f} | Std: {greek_stats[greek_name]['std']:.6f}")
            logger.info(f"    ⏱ Time for {greek_name}: {iter_time:.2f}s | ETA remaining: {remaining:.2f}s")
            
            # DIAGNOSTIC: Check for issues
            if greek_stats[greek_name]['nan_count'] > 0:
                logger.warning(f"    ⚠ WARNING: {greek_stats[greek_name]['nan_count']} NaN values in {greek_name}!")
            if greek_stats[greek_name]['inf_count'] > 0:
                logger.warning(f"    ⚠ WARNING: {greek_stats[greek_name]['inf_count']} Inf values in {greek_name}!")
            
            zero_pct = 100 * greek_stats[greek_name]['zero_count'] / greek_stats[greek_name]['total_count']
            if zero_pct > 50:
                logger.warning(f"    ⚠ WARNING: {zero_pct:.1f}% zero values in {greek_name}!")
            
            # DIAGNOSTIC: Breach vs non-breach comparison
            if n_breached > 0 and (env.M - n_breached) > 0:
                greek_breached = greek_np[breach_mask, :].mean()
                greek_not_breached = greek_np[~breach_mask, :].mean()
                logger.info(f"    ✓ Mean (breached): {greek_breached:.6f} | Mean (not breached): {greek_not_breached:.6f}")

            pbar.set_postfix_str(f"{idx}/{len(greek_names)} done | ETA {remaining:.1f}s")
            pbar.update(1)
    
    total_time = time.time() - start_time
    logger.info(f"  → Portfolio Greeks computation complete in {total_time:.2f}s")

    # ========== STEP 3: Solve for Optimal Hedge Positions ==========
    logger.info(f"STEP 3/5: Solving for optimal hedge positions using linear algebra")
    HN_positions_all = env.compute_hn_option_positions(S_traj, portfolio_greeks)
    logger.info(f"  → Hedge positions computed: shape {HN_positions_all.shape}")
    
    # DIAGNOSTIC: Analyze hedge positions
    positions_np = HN_positions_all.cpu().numpy()
    logger.info(f"  → Position diagnostics:")
    for inst_idx in range(n_hedging_instruments):
        inst_type = 'Stock' if inst_idx == 0 else f'Option {inst_idx}'
        pos = positions_np[:, :, inst_idx]
        logger.info(f"    {inst_type}: Range=[{pos.min():.4f}, {pos.max():.4f}], Mean={pos.mean():.4f}, Std={pos.std():.4f}")
        
        nan_count = np.isnan(pos).sum()
        inf_count = np.isinf(pos).sum()
        if nan_count > 0:
            logger.warning(f"    ⚠ {inst_type}: {nan_count} NaN values!")
        if inf_count > 0:
            logger.warning(f"    ⚠ {inst_type}: {inf_count} Inf values!")

    # ========== STEP 4: Simulate Hedge Strategy ==========
    logger.info(f"STEP 4/5: Simulating practitioner hedge strategy across all paths")
    _, trajectories_hn = env.simulate_full_trajectory(HN_positions_all, O_traj)
    logger.info(f"  → Simulation complete")

    # ========== STEP 5: Compute Terminal Hedge Errors with Diagnostics ==========
    logger.info(f"STEP 5/5: Computing terminal hedge errors")
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
    
    # DIAGNOSTIC: Payoff statistics
    payoff_np = payoff.cpu().numpy()
    logger.info(f"  → Payoff: Range=[{payoff_np.min():.4f}, {payoff_np.max():.4f}], Mean={payoff_np.mean():.4f}")
    logger.info(f"  → Paths ITM: {(payoff_np > 0).sum()}/{env.M}")

    terminal_value_hn = trajectories_hn['B'][:, -1] + HN_positions_all[:, -1, 0] * S_final
    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
        option_contrib = HN_positions_all[:, -1, i] * O_traj[i - 1][:, -1]
        terminal_value_hn += option_contrib
        # DIAGNOSTIC: Option contribution
        option_contrib_np = option_contrib.cpu().numpy()
        logger.info(f"  → Option {i} contribution: Range=[{option_contrib_np.min():.4f}, {option_contrib_np.max():.4f}]")

    terminal_hedge_error_hn = (terminal_value_hn - env.side * payoff).cpu().detach().numpy()
    
    # DIAGNOSTIC: Detailed error statistics
    logger.info(f"  ✓ Terminal hedge error computed")
    logger.info(f"    Mean error: {terminal_hedge_error_hn.mean():.6f}")
    logger.info(f"    Std error: {terminal_hedge_error_hn.std():.6f}")
    logger.info(f"    Min error: {terminal_hedge_error_hn.min():.6f}")
    logger.info(f"    Max error: {terminal_hedge_error_hn.max():.6f}")
    logger.info(f"    Median error: {np.median(terminal_hedge_error_hn):.6f}")
    
    # Check for extreme errors
    extreme_threshold = 3 * terminal_hedge_error_hn.std()
    extreme_errors = np.abs(terminal_hedge_error_hn) > extreme_threshold
    if extreme_errors.sum() > 0:
        logger.warning(f"    ⚠ {extreme_errors.sum()} paths with extreme errors (>3σ)")
        extreme_indices = np.where(extreme_errors)[0]
        logger.warning(f"    ⚠ Example extreme error paths: {extreme_indices[:5].tolist()}")
    
    # Performance metrics
    mse = (terminal_hedge_error_hn ** 2).mean()
    smse = mse / (env.S0 ** 2)
    cvar_95 = np.mean(np.sort(terminal_hedge_error_hn ** 2)[-int(0.05 * env.M):])
    
    logger.info(f"    MSE: {mse:.6f}")
    logger.info(f"    SMSE: {smse:.8f}")
    logger.info(f"    CVaR95: {cvar_95:.6f}")
    
    # DIAGNOSTIC: Error breakdown by breach status
    if n_breached > 0 and (env.M - n_breached) > 0:
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC: ERROR BREAKDOWN BY BREACH STATUS")
        logger.info("=" * 80)
        
        errors_breached = terminal_hedge_error_hn[breach_mask]
        errors_not_breached = terminal_hedge_error_hn[~breach_mask]
        
        mse_breached = (errors_breached ** 2).mean()
        mse_not_breached = (errors_not_breached ** 2).mean()
        
        logger.info(f"BREACHED PATHS ({n_breached} paths, {breach_pct:.1f}%):")
        logger.info(f"  Mean error: {errors_breached.mean():.6f}")
        logger.info(f"  Std error: {errors_breached.std():.6f}")
        logger.info(f"  MSE: {mse_breached:.6f}")
        
        logger.info(f"NON-BREACHED PATHS ({env.M - n_breached} paths, {100-breach_pct:.1f}%):")
        logger.info(f"  Mean error: {errors_not_breached.mean():.6f}")
        logger.info(f"  Std error: {errors_not_breached.std():.6f}")
        logger.info(f"  MSE: {mse_not_breached:.6f}")
        
        # Identify dominant error source
        if mse_breached > 5 * mse_not_breached:
            logger.warning("⚠ CRITICAL: MSE dominated by BREACHED paths!")
            logger.warning("⚠ Issue likely in vanilla option fallback after breach")
            logger.warning("⚠ Check barrier_wrapper.py Greeks after breach")
        elif mse_not_breached > 5 * mse_breached:
            logger.warning("⚠ CRITICAL: MSE dominated by NON-BREACHED paths!")
            logger.warning("⚠ Issue likely in barrier option neural network pricing/Greeks")
            logger.warning("⚠ Check barrier.py Greeks before breach")
        else:
            logger.info("✓ Error contributions are balanced between breached and non-breached paths")
    
    logger.info("=" * 80)
    logger.info("ANALYTICAL HEDGE COMPUTATION COMPLETE")
    logger.info("=" * 80)

    return HN_positions_all, trajectories_hn, terminal_hedge_error_hn


def compute_rl_metrics(
    env: Any,
    RL_positions: torch.Tensor,
    trajectories: Dict[str, torch.Tensor],
    O_traj: Dict[int, torch.Tensor]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute RL strategy metrics.
    
    Works with any derivative type.
    
    Args:
        env: Hedging environment
        RL_positions: [M, N+1, n_instruments] RL positions
        trajectories: Dict with simulation results
        O_traj: Option price trajectories
        
    Returns:
        Tuple of (terminal_hedge_error, metrics_dict)
    """
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
    """
    Create comprehensive visualization of episode results.
    
    This function is now derivative-agnostic and works with vanilla options,
    barrier options, and any future derivative type that implements the
    standard Greek interface.
    
    Args:
        episode: Episode number
        metrics: Dictionary containing training metrics and trajectories
        config: Configuration dictionary
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
        
        n_inst = config["instruments"]["n_hedging_instruments"]
        path_idx = config["output"]["sample_path_index"]
        
        # Determine derivative type for logging
        derivative_type = type(env.derivative).__name__
        logger.info(f"Plotting results for {derivative_type}")
        
        # ===== DIAGNOSTIC VALIDATION =====
        logger.info("=" * 80)
        logger.info("O_traj Diagnostic Info")
        logger.info("=" * 80)
        logger.info(f"Type of O_traj: {type(O_traj)}")
        logger.info(f"Number of entries in O_traj: {len(O_traj)}")
        
        # Convert O_traj to dict if it's a list
        if isinstance(O_traj, list):
            logger.info("O_traj is a list, converting to dict with integer keys")
            O_traj = {i: tensor for i, tensor in enumerate(O_traj)}
        
        for key in O_traj.keys():
            value = O_traj[key]
            logger.info(f"O_traj[{key}]: shape={value.shape}, dtype={value.dtype}, numel={value.numel()}")
        logger.info(f"Number of hedging instruments: {n_inst}")
        logger.info(f"Instrument maturities: {env.instrument_maturities}")
        logger.info(f"Expected hedging options: {n_inst - 1}")
        logger.info("=" * 80)
        
        # Validate O_traj
        expected_instruments = n_inst - 1  # Excluding stock
        if len(O_traj) != expected_instruments:
            logger.error(f"O_traj has {len(O_traj)} entries but expected {expected_instruments}")
            logger.error("Cannot proceed with plotting due to instrument count mismatch")
            return
        
        # Check all tensors have consistent shapes
        for key, value in O_traj.items():
            if value.numel() == 0 or len(value.shape) == 0:
                logger.error(f"O_traj[{key}] is empty or scalar: shape={value.shape}")
                logger.error("Cannot proceed with plotting due to empty option trajectory")
                return
            if value.shape[0] != S_traj.shape[0]:
                logger.error(f"O_traj[{key}] shape mismatch: {value.shape[0]} paths vs {S_traj.shape[0]} stock paths")
                logger.error("Cannot proceed with plotting due to shape mismatch")
                return
        
        # Compute RL metrics
        logger.info("Computing RL metrics...")
        terminal_hedge_error_rl, rl_metrics = compute_rl_metrics(
            env, RL_positions, trajectories, O_traj
        )
        
        # Compute practitioner benchmark (now with enhanced diagnostics)
        logger.info("Computing practitioner benchmark...")
        HN_positions_all, trajectories_hn, terminal_hedge_error_hn = \
            compute_practitioner_benchmark(env, S_traj, O_traj, n_inst)
        
        # Compute HN metrics
        mse_hn = float(np.mean(terminal_hedge_error_hn ** 2))
        smse_hn = mse_hn / (env.S0 ** 2)
        cvar_95_hn = float(np.mean(np.sort(terminal_hedge_error_hn ** 2)[-int(0.05 * env.M):]))
        
        # Create figure with subplots
        logger.info("Creating figure and subplots...")
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        time_steps = np.arange(env.N + 1)
        
        # Extract sample paths for plotting
        rl_positions_sample = RL_positions[path_idx].cpu().detach().numpy()
        hn_positions_sample = HN_positions_all[path_idx].cpu().detach().numpy()
        
        # ===== PLOT 1: Stock Delta Comparison =====
        logger.info("Creating Plot 1: Stock Delta Comparison...")
        axes[0, 0].plot(time_steps, rl_positions_sample[:, 0], label='RL Delta',
                        linewidth=2, color='tab:blue')
        axes[0, 0].plot(time_steps, hn_positions_sample[:, 0], label='Practitioner Delta',
                        linewidth=2, linestyle='--', alpha=0.8, color='tab:orange')
        axes[0, 0].set_xlabel("Time Step", fontsize=11)
        axes[0, 0].set_ylabel("Delta", fontsize=11)
        axes[0, 0].set_title(f"Stock Delta: Practitioner vs RL (Path {path_idx})", fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # ===== PLOT 2: Option Positions Comparison =====
        logger.info("Creating Plot 2: Option Positions Comparison...")
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
        
        # ===== PLOT 3: Stock Price Trajectory with Derivative Price =====
        logger.info("Creating Plot 3: Stock Price Trajectory...")
        ax1 = axes[1, 0]
        color_stock = 'tab:green'
        ax1.plot(time_steps, S_traj[path_idx].cpu().detach().numpy(),
                 label='Stock Price', color=color_stock, linewidth=2)
        ax1.axhline(y=env.K, color='r', linestyle='--', label='Strike', alpha=0.7)
        
        # Add barrier level if hedging a barrier option
        if hasattr(env.derivative, 'barrier_level'):
            ax1.axhline(y=env.derivative.barrier_level, color='purple', 
                       linestyle=':', label='Barrier', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel("Time Step", fontsize=11)
        ax1.set_ylabel("Stock Price", fontsize=11, color=color_stock)
        ax1.tick_params(axis='y', labelcolor=color_stock)
        ax1.grid(True, alpha=0.3)
        
        # Plot derivative price on secondary y-axis with independent scale
        ax2 = ax1.twinx()
        color_derivative = 'tab:blue'
        ax2.plot(time_steps, V_traj[path_idx].cpu().detach().numpy(),
                 label=f'{derivative_type} Price', color=color_derivative, 
                 linewidth=2, alpha=0.8)
        ax2.set_ylabel(f"{derivative_type} Price", fontsize=11, color=color_derivative)
        ax2.tick_params(axis='y', labelcolor=color_derivative)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')
        
        ax1.set_title(f"Stock & {derivative_type} Price (Path {path_idx})", fontsize=12)
        
        # ===== PLOT 4: Hedging Instrument Prices =====
        logger.info("Creating Plot 4: Hedging Instrument Prices...")
        if n_inst >= 2:
            for i in range(1, n_inst):
                opt_idx = i - 1  # O_traj is 0-indexed for options only
                if opt_idx not in O_traj:
                    logger.warning(f"Option index {opt_idx} not found in O_traj, skipping")
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
        
        # ===== PLOT 5: Position Difference (RL - Practitioner) =====
        logger.info("Creating Plot 5: Position Difference...")
        delta_diff = rl_positions_sample[:, 0] - hn_positions_sample[:, 0]
        axes[2, 0].plot(time_steps, delta_diff, color='tab:red', linewidth=2)
        axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[2, 0].set_xlabel("Time Step", fontsize=11)
        axes[2, 0].set_ylabel("Delta Difference", fontsize=11)
        axes[2, 0].set_title(f"RL Delta - Practitioner Delta (Path {path_idx})", fontsize=12)
        axes[2, 0].grid(True, alpha=0.3)
        
        # ===== PLOT 6: Terminal Error Distribution =====
        logger.info("Creating Plot 6: Terminal Error Distribution...")
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
        
        # Adjust layout and save
        logger.info("Saving plot...")
        fig.tight_layout()
        save_path = config["output"]["plot_save_path"].format(
            n_inst=n_inst,
            episode=episode
        )
        plt.savefig(save_path, dpi=config["output"]["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {save_path}")
        
        # Log detailed statistics
        delta_mae = np.mean(np.abs(delta_diff))
        delta_rmse = np.sqrt(np.mean(delta_diff ** 2))
        
        logger.info(
            f"Path {path_idx} Delta Statistics - MAE: {delta_mae:.6f} | RMSE: {delta_rmse:.6f}"
        )
        
        if n_inst >= 2:
            for i in range(1, n_inst):
                position_diff = rl_positions_sample[:, i] - hn_positions_sample[:, i]
                position_mae = np.mean(np.abs(position_diff))
                position_rmse = np.sqrt(np.mean(position_diff ** 2))
                logger.info(
                    f"Path {path_idx} Instrument {i} Position Statistics - "
                    f"MAE: {position_mae:.6f} | RMSE: {position_rmse:.6f}"
                )
        
        logger.info(
            f"RL Performance: MSE={rl_metrics['mse']:.6f} | "
            f"SMSE={rl_metrics['smse']:.6f} | CVaR95={rl_metrics['cvar_95']:.6f}"
        )
        logger.info(
            f"Practitioner Performance: MSE={mse_hn:.6f} | "
            f"SMSE={smse_hn:.6f} | CVaR95={cvar_95_hn:.6f}"
        )
        logger.info(f"Derivative Type: {derivative_type}")
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PLOTTING FAILED WITH ERROR")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        logger.error("=" * 80)
        logger.error("Attempting to provide diagnostic information...")
        
        try:
            # Try to provide useful diagnostic info
            env = metrics.get('env')
            O_traj = metrics.get('O_traj')
            
            if env is not None:
                logger.error(f"Environment info:")
                logger.error(f"  - N (timesteps): {env.N}")
                logger.error(f"  - M (paths): {env.M}")
                logger.error(f"  - Instrument maturities: {env.instrument_maturities}")
                logger.error(f"  - Instrument types: {env.instrument_types}")
                
            if O_traj is not None:
                logger.error(f"O_traj info:")
                logger.error(f"  - Type: {type(O_traj)}")
                logger.error(f"  - Length: {len(O_traj)}")
                if isinstance(O_traj, dict):
                    for key, value in O_traj.items():
                        logger.error(f"  - O_traj[{key}]: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(O_traj, list):
                    for i, value in enumerate(O_traj):
                        logger.error(f"  - O_traj[{i}]: shape={value.shape}, dtype={value.dtype}")
        except Exception as diag_error:
            logger.error(f"Could not provide diagnostics: {diag_error}")
        
        logger.error("=" * 80)
        logger.error("Plotting aborted. Check logs above for details.")
        logger.error("=" * 80)
