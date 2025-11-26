"""
Main training script for GARCH-based option hedging with RL.

This script loads configuration from YAML, initializes all components,
and runs the training loop. It replaces the train_garch function with
a more modular, configuration-driven approach.

Supports vanilla European, barrier, American, and Asian options.
Automatically detects and uses CUDA if available, otherwise falls back to CPU.

NEW: Supports configurable risk measures (MSE, SMSE, CVaR, Variance, MAE)
     and soft error constraint tracking.

Usage:
    python train.py --config cfgs/config_vanilla_2inst.yaml
    python train.py --config cfgs/config_barrier_2inst.yaml
    python train.py --config cfgs/config_american_1inst.yaml
    python train.py --config cfgs/config_asian_2inst.yaml
    python train.py --config cfgs/config.yaml --load-model models/uniform/GARCHLSTMDG.pth --inference-only
"""

import yaml
import torch
import numpy as np
import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.agents.policy_net_garch_SCRMFG import (
    PolicyNetGARCH, 
    HedgingEnvGARCH,
    compute_loss_with_soft_constraint
)
from src.option_greek.precompute import create_precomputation_manager_from_config
from src.visualization.plot_results import compute_rl_metrics
from derivative_factory import setup_derivatives_from_precomputed


logger = logging.getLogger(__name__)


def auto_detect_device() -> str:
    """Auto-detect device: use CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"CUDA is available! Detected GPU: {device_name}")
        return "cuda"
    else:
        logging.info("CUDA not available. Using CPU.")
        return "cpu"


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging based on configuration."""
    log_level = getattr(logging, config["logging"]["level"])
    log_file = config["logging"].get("log_file")
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config name without 'config' prefix and without extension
    # e.g., 'configABCDEF.yaml' -> 'ABCDEF'
    config_filename = os.path.basename(config_path)
    config_name = config_filename.replace('.yaml', '').replace('.yml', '')
    
    # Remove 'config' prefix if present (case-insensitive)
    if config_name.lower().startswith('config'):
        config_name = config_name[6:]  # Remove first 6 characters ('config')
    
    config["config_name"] = config_name
    
    logging.info(f"Loaded configuration from {config_path}")
    logging.info(f"Config name extracted: {config_name}")
    return config


def get_transaction_costs(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Get transaction costs from config.
    
    Supports two formats:
    1. Simple flag: transaction_costs: {TC: true/false}
    2. Detailed: transaction_costs: {stock: 0.0001, vanilla_option: 0.001, ...}
    """
    tc_config = config.get("transaction_costs", {})
    
    # Check if using simple TC flag format
    if "TC" in tc_config:
        if tc_config["TC"]:
            # Use default transaction costs
            return {
                'stock': 0.0001,
                'vanilla_option': 0.001,
                'barrier_option': 0.002,
                'american_option': 0.0015,
                'asian_option': 0.0015
            }
        else:
            # All zero transaction costs
            return {
                'stock': 0.0,
                'vanilla_option': 0.0,
                'barrier_option': 0.0,
                'american_option': 0.0,
                'asian_option': 0.0
            }
    else:
        # Use detailed transaction costs with defaults for missing values
        defaults = {
            'stock': 0.0001,
            'vanilla_option': 0.001,
            'barrier_option': 0.002,
            'american_option': 0.0015,
            'asian_option': 0.0015
        }
        return {key: tc_config.get(key, defaults[key]) for key in defaults}


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters including risk measure and soft constraint."""
    # Check for mode
    mode = config["instruments"].get("mode", "static")
    
    if mode == "floating_grid":
        # Floating grid validation
        grid_config = config["instruments"].get("floating_grid", {})
        if not grid_config.get("enabled", False):
            raise ValueError("floating_grid mode selected but floating_grid.enabled is not true")
        
        moneyness_levels = grid_config.get("moneyness_levels", [])
        maturity_days = grid_config.get("maturity_days", [])
        
        if not moneyness_levels or not maturity_days:
            raise ValueError("floating_grid requires moneyness_levels and maturity_days")
        
        grid_size = len(moneyness_levels) * len(maturity_days)
        expected_n_inst = 1 + grid_size  # 1 stock + grid_size options
        actual_n_inst = config["instruments"]["n_hedging_instruments"]
        
        if expected_n_inst != actual_n_inst:
            raise ValueError(
                f"Floating grid expects n_hedging_instruments={expected_n_inst} "
                f"(1 stock + {grid_size} options), got {actual_n_inst}"
            )
        
        logging.info(f"Floating grid mode validated: {len(moneyness_levels)} moneyness × {len(maturity_days)} maturities = {grid_size} options")
    
    else:
        # Static mode validation
        n_inst = config["instruments"]["n_hedging_instruments"]
        if n_inst < 1 or n_inst > 31:
            raise ValueError(f"n_hedging_instruments must be 1-31, got {n_inst}")
        
        if n_inst > 1:
            n_strikes = len(config["instruments"]["strikes"])
            n_types = len(config["instruments"]["types"])
            n_maturities = len(config["instruments"]["maturities"])
            
            if n_strikes != n_inst - 1:
                raise ValueError(
                    f"strikes must have length {n_inst - 1} (excluding stock), got {n_strikes}"
                )
            if n_types != n_inst - 1:
                raise ValueError(
                    f"types must have length {n_inst - 1} (excluding stock), got {n_types}"
                )
            if n_maturities != n_inst - 1:
                raise ValueError(
                    f"maturities must have length {n_inst - 1} (excluding stock), got {n_maturities}"
                )
        
        valid_types = ["call", "put"]
        for opt_type in config["instruments"]["types"]:
            if opt_type not in valid_types:
                raise ValueError(f"Invalid option type: {opt_type}")
    
    hedged_cfg = config["hedged_option"]
    deriv_type = hedged_cfg["type"].lower()
    
    # Extended validation for American and Asian options
    if deriv_type not in ["vanilla", "barrier", "american", "asian"]:
        raise ValueError(f"Invalid hedged_option type: {deriv_type}")
    
    if hedged_cfg["option_type"] not in ["call", "put"]:
        raise ValueError(
            f"Invalid hedged_option option_type: {hedged_cfg['option_type']}"
        )
    
    if hedged_cfg["side"] not in ["long", "short"]:
        raise ValueError(
            f"Invalid hedged_option side: {hedged_cfg['side']}"
        )
    
    # Validate that American and Asian options have model_path if needed
    if deriv_type in ["american", "asian"] and "model_path" not in hedged_cfg:
        logging.warning(f"{deriv_type.capitalize()} option specified but no model_path provided in config")
    
    # NEW: Validate risk measure configuration
    if "risk_measure" in config:
        risk_config = config["risk_measure"]
        valid_risk_measures = ["mse", "smse", "cvar", "var", "mae"]
        risk_type = risk_config.get("type", "mse")
        
        if risk_type not in valid_risk_measures:
            raise ValueError(f"Invalid risk_measure type: {risk_type}. Must be one of {valid_risk_measures}")
        
        # CVaR and VaR require alpha parameter
        if risk_type in ["cvar", "var"] and "alpha" not in risk_config:
            raise ValueError(f"risk_measure type '{risk_type}' requires 'alpha' parameter")
        
        # Warn if alpha is provided for non-CVaR/VaR measures
        if risk_type not in ["cvar", "var"] and "alpha" in risk_config:
            logging.warning(f"alpha parameter is ignored for risk_measure type '{risk_type}' (only used for 'cvar' or 'var')")
        
        # Warning for VaR (non-convex)
        if risk_type == "var":
            logging.warning("VaR is NOT a convex or coherent risk measure. Consider using CVaR instead for better optimization properties.")
    
    # NEW: Validate soft constraint configuration
    if "soft_constraint" in config:
        constraint_config = config["soft_constraint"]
        if "enabled" not in constraint_config:
            logging.warning("soft_constraint section found but 'enabled' not specified, defaulting to false")
        if "lambda" not in constraint_config:
            logging.warning("soft_constraint section found but 'lambda' not specified, defaulting to 0.0")
    
    logging.info("Configuration validation passed")

def create_policy_network(config: Dict[str, Any], device: torch.device) -> PolicyNetGARCH:
    """Create and initialize policy network."""
    model_config = config["model"]
    
    # Get use_action_recurrence from config, default to False
    use_action_recurrence = model_config.get("use_action_recurrence", False)
    
    policy_net = PolicyNetGARCH(
        obs_dim=model_config["obs_dim"],
        hidden_size=model_config["hidden_size"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        num_lstm_layers=model_config["lstm_layers"],
        num_fc_layers=model_config["num_layers"],
        use_action_recurrence=use_action_recurrence  # ← ADD THIS
    ).to(device)
    
    logging.info(
        f"Created policy network with {model_config['hidden_size']} hidden units, "
        f"{model_config['num_layers']} FC layers, "
        f"action_recurrence={use_action_recurrence}"  # ← ADD THIS
    )
    
    return policy_net

def create_optimizer(
    policy_net: PolicyNetGARCH,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Create optimizer for policy network."""
    train_config = config["training"]
    
    if train_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            policy_net.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    elif train_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    logging.info(
        f"Created {train_config['optimizer']} optimizer with "
        f"lr={train_config['learning_rate']}, weight_decay={train_config['weight_decay']}"
    )
    
    return optimizer


def train_episode(
    episode: int,
    config: Dict[str, Any],
    policy_net: PolicyNetGARCH,
    optimizer: torch.optim.Optimizer,
    hedged_derivative,
    hedging_derivatives,
    HedgingSim,
    device: torch.device,
    precomputation_manager
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
    alpha = risk_config.get("alpha", None)  # Only needed for CVaR
    lambda_constraint = constraint_config.get("lambda", 0.0) if constraint_config.get("enabled", False) else 0.0
    
    # Get sparsity penalty for floating grid
    lambda_sparsity = 0.0
    if is_floating_grid:
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
        hedging_derivatives=None if is_floating_grid else hedging_derivatives,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        transaction_costs=transaction_costs,
        grid_config=config if is_floating_grid else None,
        precomputation_manager=precomputation_manager
    )

    env.reset()
    
    S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
        env.simulate_trajectory_and_get_observations(policy_net)
    
    terminal_errors, trajectories = env.simulate_full_trajectory(RL_positions, O_traj)
    
    optimizer.zero_grad()
    
    # NEW: Compute loss with configurable risk measure and soft constraint
    total_loss, risk_loss, constraint_penalty, sparsity_penalty = compute_loss_with_soft_constraint(
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
    
    # NEW: Enhanced logging with risk measure and constraint info
    log_msg = (
        f"Episode {episode} | "
        f"Risk: {risk_measure.upper()} | "
        f"Total Loss: {total_loss.item():.6f} | "
        f"Risk Loss: {risk_loss.item():.6f}"
    )
    
    if lambda_constraint > 0:
        avg_violation = trajectories['soft_constraint_violations'].mean().item()
        log_msg += (
            f" | Constraint Penalty: {constraint_penalty.item():.6f} "
            f"(λ={lambda_constraint}) | "
            f"Avg Violation: {avg_violation:.6f}"
        )
    
    if lambda_sparsity > 0:
        log_msg += f" | Sparsity Penalty: {sparsity_penalty.item():.6f} (λ={lambda_sparsity})"
    
    log_msg += f" | Final Reward: {final_reward:.6f}"
    
    # Add ledger info for floating grid
    if is_floating_grid and "ledger_size_trajectory" in trajectories:
        avg_ledger = np.mean(trajectories["ledger_size_trajectory"])
        max_ledger = np.max(trajectories["ledger_size_trajectory"])
        log_msg += f" | Ledger (Avg/Max): {avg_ledger:.1f}/{max_ledger:.0f}"
    
    logging.info(log_msg)
    
    return {
        "episode": episode,
        "loss": total_loss.item(),
        "risk_loss": risk_loss.item(),
        "constraint_penalty": constraint_penalty.item(),
        "sparsity_penalty": sparsity_penalty.item(),
        "reward": final_reward,
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env,
        "risk_measure": risk_measure,
        "lambda_constraint": lambda_constraint
    }


def save_checkpoint(
    policy_net: PolicyNetGARCH,
    config: Dict[str, Any],
    episode: int,
    config_name: str
) -> None:
    """Save model checkpoint with config name in filename."""
    # Determine hedging instrument name based on n_hedging_instruments
    n_inst = config["instruments"]["n_hedging_instruments"]
    
    if n_inst == 1:
        instrument_name = "stock_only"
    elif n_inst == 2:
        instrument_name = "2inst"
    elif n_inst == 3:
        instrument_name = "3inst"
    elif n_inst == 4:
        instrument_name = "4inst"
    else:
        instrument_name = f"{n_inst}inst"
    
    # Create checkpoint directory: models/{instrument_name}/checkpoint
    checkpoint_dir = os.path.join("models", instrument_name, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename: {config_name}.pth
    checkpoint_filename = f"{config_name}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    torch.save(policy_net.state_dict(), checkpoint_path)
    logging.info(f"Checkpoint saved at episode {episode}: {checkpoint_path}")


def run_inference(
    config: Dict[str, Any],
    policy_net: PolicyNetGARCH,
    hedged_derivative,
    hedging_derivatives,
    HedgingSim,
    device: torch.device
) -> None:
    """Run inference with a pretrained model and generate visualizations."""
    logging.info("Starting inference with pretrained model...")
    
    policy_net.eval()
    
    hedged_cfg = config["hedged_option"]
    mode = config["instruments"].get("mode", "static")
    is_floating_grid = (mode == "floating_grid")
    
    # Get transaction costs from config
    transaction_costs = get_transaction_costs(config)
    
    # FIX: Convert K and S0 to tensors on device BEFORE passing to HedgingSim
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
        seed=config["training"]["seed"]
    )
    
    env = HedgingEnvGARCH(
       sim=sim,
       derivative=hedged_derivative,
       hedging_derivatives=None if is_floating_grid else hedging_derivatives,
       garch_params=config["garch"],
       n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
       dt_min=config["environment"]["dt_min"],
       device=str(device),
       transaction_costs=transaction_costs,
       grid_config=config if is_floating_grid else None,
       precomputation_manager=hedging_derivatives.precomp_manager  # ← ADD THIS LINE
    )
    
    env.reset()
     
    with torch.no_grad():
        S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
            env.simulate_trajectory_and_get_observations(policy_net)
        
        terminal_errors, trajectories = env.simulate_full_trajectory(RL_positions, O_traj)
    
    terminal_hedge_error_rl, rl_metrics = compute_rl_metrics(
        env, RL_positions, trajectories, O_traj
    )
    
    logging.info(
        f"Inference Results - MSE: {rl_metrics['mse']:.6f} | "
        f"SMSE: {rl_metrics['smse']:.6f} | CVaR95: {rl_metrics['cvar_95']:.6f}"
    )
    
    # NEW: Also compute and log the configured risk measure
    risk_config = config.get("risk_measure", {"type": "mse"})
    constraint_config = config.get("soft_constraint", {"enabled": False, "lambda": 0.0})
    
    risk_measure = risk_config.get("type", "mse")
    alpha = risk_config.get("alpha", None)
    lambda_constraint = constraint_config.get("lambda", 0.0) if constraint_config.get("enabled", False) else 0.0
    lambda_sparsity = 0.0
    if is_floating_grid:
        lambda_sparsity = config["instruments"]["floating_grid"].get("sparsity_penalty", 0.0)
    
    total_loss, risk_loss, constraint_penalty, sparsity_penalty = compute_loss_with_soft_constraint(
        terminal_errors,
        trajectories,
        risk_measure=risk_measure,
        alpha=alpha,
        lambda_constraint=lambda_constraint,
        lambda_sparsity=lambda_sparsity
    )
    
    logging.info(
        f"Configured Risk Measure ({risk_measure.upper()}): {risk_loss.item():.6f} | "
        f"Constraint Penalty: {constraint_penalty.item():.6f}"
    )
    
    if lambda_sparsity > 0:
        logging.info(f"Sparsity Penalty: {sparsity_penalty.item():.6f}")
    
    # ============================================================
    # CONSTRUCT PROPER SAVE PATH
    # ============================================================
    # Get config name (without 'config' prefix and without extension)
    config_name = config.get("config_name", "model")
    
    # Determine derivative type from config name prefix
    if config_name.startswith('AM'):
        derivative_folder = "american"
    elif config_name.startswith('B'):
        derivative_folder = "barrier"
    elif config_name.startswith('D'):
        derivative_folder = "vanilla"
    else:
        # Fallback: use the hedged_option type from config
        derivative_folder = config["hedged_option"]["type"].lower()
    
    # Determine instrument folder name
    if is_floating_grid:
        instrument_folder = "floating_grid"
    else:
        n_inst = config["instruments"]["n_hedging_instruments"]
        if n_inst == 1:
            instrument_folder = "1inst"
        elif n_inst == 2:
            instrument_folder = "2inst"
        elif n_inst == 3:
            instrument_folder = "3inst"
        elif n_inst == 4:
            instrument_folder = "4inst"
        else:
            instrument_folder = f"{n_inst}inst"
    
    # Determine TC folder based on config name ending
    # Config names ending in 'X' → NoTC
    # Config names NOT ending in 'X' → TC
    if config_name.endswith('X'):
        tc_folder = "NoTC"
    else:
        tc_folder = "TC"
    
    # Get risk measure folder name
    risk_measure_folder = risk_measure.upper()
    
    # Construct path: visual_results/{derivative}/{instrument}/{tc}/{risk_measure}/{config_name}.png
    save_dir = os.path.join("visual_results", derivative_folder, instrument_folder, tc_folder, risk_measure_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{config_name}.png")
    
    logging.info(f"Plot will be saved to: {save_path}")
    
    # Override the config's plot_save_path with our constructed path
    if "output" not in config:
        config["output"] = {}
    config["output"]["plot_save_path"] = save_path
    
    metrics = {
        "episode": 0,
        "loss": total_loss.item(),
        "risk_loss": risk_loss.item(),
        "constraint_penalty": constraint_penalty.item(),
        "sparsity_penalty": sparsity_penalty.item(),
        "reward": -float(total_loss.item()),
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env,
        "risk_measure": risk_measure, 
        "lambda_constraint": lambda_constraint
    }
    
    try:
        from src.visualization.plot_results import plot_episode_results
        plot_episode_results(episode=0, metrics=metrics, config=config)
        logging.info(f"Inference plots saved to: {save_path}")
    except Exception as e:
        logging.warning(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

def train(
    config: Dict[str, Any],
    HedgingSim,
    hedged_derivative,
    hedging_derivatives,
    visualize: bool = True,
    initial_model: Optional[PolicyNetGARCH] = None,
    config_name: str = "config",
    precomputation_manager: Any=None
) -> PolicyNetGARCH:
    """Main training loop."""
    
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(config["training"]["device"])
    logging.info(f"Using device: {device}")
    
    # NEW: Log risk measure and soft constraint configuration
    risk_config = config.get("risk_measure", {"type": "mse"})
    constraint_config = config.get("soft_constraint", {"enabled": False, "lambda": 0.0})
    
    logging.info("=" * 70)
    logging.info("RISK MEASURE & SOFT CONSTRAINT CONFIGURATION")
    logging.info(f"Risk Measure: {risk_config.get('type', 'mse').upper()}")
    if risk_config.get('type') == 'cvar':
        logging.info(f"CVaR Alpha: {risk_config.get('alpha', 0.95)}")
    logging.info(f"Soft Constraint Enabled: {constraint_config.get('enabled', False)}")
    if constraint_config.get('enabled', False):
        logging.info(f"Constraint Lambda: {constraint_config.get('lambda', 0.0)}")
    logging.info("=" * 70)
    
    policy_net = create_policy_network(config, device)
    
    if initial_model is not None:
        policy_net.load_state_dict(initial_model.state_dict())
        logging.info("Initialized policy network from pretrained model")
    
    optimizer = create_optimizer(policy_net, config)
    
    n_episodes = config["training"]["episodes"]
    checkpoint_freq = config["training"]["checkpoint_frequency"]
    plot_freq = config["training"]["plot_frequency"]
    
    logging.info(
        f"Starting training: {n_episodes} episodes, "
        f"{config['instruments']['n_hedging_instruments']} instruments, "
        f"device={device}"
    )
    
    for episode in range(1, n_episodes + 1):
        try:
            metrics = train_episode(
                episode=episode,
                config=config,
                policy_net=policy_net,
                optimizer=optimizer,
                hedged_derivative=hedged_derivative,
                hedging_derivatives=hedging_derivatives,
                HedgingSim=HedgingSim,
                device=device,
                precomputation_manager=precomputation_manager
            )
            
            if episode % checkpoint_freq == 0:
                save_checkpoint(policy_net, config, episode, config_name)
            
            if visualize and episode % plot_freq == 0:
                try:
                    from src.visualization.plot_results import plot_episode_results
                    plot_episode_results(episode, metrics, config)
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
        
        except Exception as e:
            logging.exception(f"Error during episode {episode}: {e}")
            raise
    
    # Save final model with config name in models/{instrument_name}/checkpoint
    n_inst = config["instruments"]["n_hedging_instruments"]
    
    if n_inst == 1:
        instrument_name = "stock_only"
    elif n_inst == 2:
        instrument_name = "2inst"
    elif n_inst == 3:
        instrument_name = "3inst"
    elif n_inst == 4:
        instrument_name = "4inst"
    else:
        instrument_name = f"{n_inst}inst"
    
    final_dir = os.path.join("models", instrument_name, "checkpoint")
    final_filename = f"{config_name}.pth"
    final_path = os.path.join(final_dir, final_filename)
    
    os.makedirs(final_dir, exist_ok=True)
    torch.save(policy_net.state_dict(), final_path)
    logging.info(f"Training finished. Model saved to {final_path}")
    
    return policy_net


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train GARCH-based option hedging with RL (supports vanilla, barrier, American, and Asian options)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization during training"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference only without training"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Extract config name from the config dict (already processed in load_config)
    config_name = config.get("config_name", "model")
    
    # ============================================================
    # AUTO-DETECT DEVICE: Use CUDA if available, otherwise CPU
    # This OVERRIDES whatever device is in the config file
    # ============================================================
    auto_device = auto_detect_device()
    config["training"]["device"] = auto_device
    config["precomputation"]["device"] = auto_device
    
    setup_logging(config)
    
    logging.info("=" * 70)
    logging.info(f"CONFIGURATION: {config_name}")
    logging.info(f"DEVICE CONFIGURATION: {auto_device.upper()}")
    if auto_device == "cuda":
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logging.info("=" * 70)
    
    validate_config(config)
    
    device = torch.device(config["training"]["device"])
    logging.info(f"Using device: {device}")
    
    try:
        from src.simulation.hedging_sim import HedgingSim
    except ImportError:
        logging.error("Could not import HedgingSim. Please adjust import path.")
        sys.exit(1)
    
    hedged_type = config["hedged_option"]["type"].lower()
    
    logging.info("Starting precomputation...")
    logging.info(f"Hedged derivative type: {hedged_type}")
    
    # Precomputation is needed for:
    # 1. Vanilla options used as hedging instruments (specified in config["instruments"]["maturities"])
    # 2. Vanilla option being hedged (if hedged_type == "vanilla")
    # 3. Barrier option being hedged (needs vanilla fallback at same maturity)
    # NOTE: American and Asian options do NOT need precomputation
    
    precomputation_manager = create_precomputation_manager_from_config(config)
    logger.info(f"precomputation_manager: {precomputation_manager.maturities}")
    
    # Check if hedged derivative needs precomputation
    needs_hedged_precompute = hedged_type in ["vanilla", "barrier"]
    
    if needs_hedged_precompute:
        hedged_maturity_days = int(config["hedged_option"]["T"] * 252)
     
         # Ensure manager has a maturities list
        if not hasattr(precomputation_manager, "maturities") or precomputation_manager.maturities is None:
            precomputation_manager.maturities = []
     
         # Add only if not already present
        if hedged_maturity_days not in precomputation_manager.maturities:
             
            precomputation_manager.maturities.append(hedged_maturity_days)
     
            logging.info(
                f"{hedged_type.capitalize()} hedged derivative at maturity "
                f"{hedged_maturity_days} days requires precomputation — added to list"
            )

    
    precomputed_data = precomputation_manager.precompute_all()
    logging.info(f"Precomputation complete for maturities: {list(precomputed_data.keys())}")
     
    # NEW: Move all precomputed tensors to device
    logging.info(f"Moving precomputed coefficients to device: {device}")
    for maturity, coeff_dict in precomputed_data.items():
        for key, value in coeff_dict.items():
            if isinstance(value, torch.Tensor):
                precomputed_data[maturity][key] = value.to(device)
    logging.info(f"All precomputed data successfully moved to {device}")
    # Verify all needed maturities are precomputed
    if needs_hedged_precompute:
        hedged_maturity_days = int(config["hedged_option"]["T"] * 252)
        if hedged_maturity_days not in precomputed_data:
            logging.warning(f"Maturity {hedged_maturity_days} not in precomputed data, computing now...")
            precomputation_manager.precompute_for_maturity(hedged_maturity_days)
            precomputed_data[hedged_maturity_days] = precomputation_manager.get_precomputed_data(hedged_maturity_days)
            logging.info(f"Precomputation complete for N={hedged_maturity_days}")
    
    hedged_derivative, hedging_derivatives, passed_precomputation_manager = setup_derivatives_from_precomputed(
        config, precomputation_manager 
    )
    
    if args.inference_only and args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        policy_net = create_policy_network(config, device)
        policy_net.load_state_dict(torch.load(args.load_model, map_location=device))
        logging.info("Model loaded successfully")
        
        run_inference(
            config=config,
            policy_net=policy_net,
            hedged_derivative=hedged_derivative,
            hedging_derivatives=hedging_derivatives,
            HedgingSim=HedgingSim,
            device=device
        )
        logging.info("Inference complete!")
        return
    
    initial_model = None
    if args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        initial_model = create_policy_network(config, device)
        initial_model.load_state_dict(torch.load(args.load_model, map_location=device))
        logging.info("Model loaded - will continue training from checkpoint")
    
    policy_net = train(
        config=config,
        HedgingSim=HedgingSim,
        hedged_derivative=hedged_derivative,
        hedging_derivatives=hedging_derivatives,
        visualize=not args.no_visualize,
        initial_model=initial_model,
        config_name=config_name,
        precomputation_manager=precomputation_manager
    )
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
