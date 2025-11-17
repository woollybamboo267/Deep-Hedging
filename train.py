"""
Main training script for GARCH-based option hedging with RL.

This script loads configuration from YAML, initializes all components,
and runs the training loop. It replaces the train_garch function with
a more modular, configuration-driven approach.

Supports vanilla European, barrier, American, and Asian options.
Automatically detects and uses CUDA if available, otherwise falls back to CPU.

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
from pathlib import Path
from typing import Dict, Any, Optional

from src.agents.policy_net_garch_flexible import PolicyNetGARCH, HedgingEnvGARCH
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
    logging.info(f"Loaded configuration from {config_path}")
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
    """Validate configuration parameters."""
    n_inst = config["instruments"]["n_hedging_instruments"]
    
    if n_inst < 1 or n_inst > 4:
        raise ValueError(f"n_hedging_instruments must be 1-4, got {n_inst}")
    
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
    
    if hedged_cfg["option_type"] not in valid_types:
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
    
    logging.info("Configuration validation passed")


def create_policy_network(config: Dict[str, Any], device: torch.device) -> PolicyNetGARCH:
    """Create and initialize policy network."""
    model_config = config["model"]
    
    policy_net = PolicyNetGARCH(
        obs_dim=model_config["obs_dim"],
        hidden_size=model_config["hidden_size"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        num_layers=model_config["num_layers"]
    ).to(device)
    
    logging.info(
        f"Created policy network with {model_config['hidden_size']} hidden units, "
        f"{model_config['num_layers']} FC layers"
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
    device: torch.device
) -> Dict[str, Any]:
    """Train for a single episode."""
    
    hedged_cfg = config["hedged_option"]
    
    # Get transaction costs from config
    transaction_costs = get_transaction_costs(config)
    
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
        hedging_derivatives=hedging_derivatives,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        transaction_costs=transaction_costs
    )
    
    env.reset()
    
    S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
        env.simulate_trajectory_and_get_observations(policy_net)
    
    terminal_errors, trajectories = env.simulate_full_trajectory(RL_positions, O_traj)
    
    optimizer.zero_grad()
    loss = torch.abs(terminal_errors).mean()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=config["training"]["gradient_clip_max_norm"]
    )
    
    optimizer.step()
    
    if torch.isnan(loss) or torch.isinf(loss):
        logging.error("Loss became NaN/Inf")
        raise RuntimeError("Loss became NaN/Inf")
    
    final_reward = -float(loss.item())
    
    logging.info(
        f"Episode {episode} | Final Reward: {final_reward:.6f} | "
        f"Total Loss: {loss.item():.6f}"
    )
    
    return {
        "episode": episode,
        "loss": loss.item(),
        "reward": final_reward,
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env
    }


def save_checkpoint(
    policy_net: PolicyNetGARCH,
    config: Dict[str, Any],
    episode: int
) -> None:
    """Save model checkpoint."""
    checkpoint_path = config["output"]["checkpoint_path"]
    
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
    
    # Get transaction costs from config
    transaction_costs = get_transaction_costs(config)
    
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
        seed=config["training"]["seed"]
    )
    
    env = HedgingEnvGARCH(
        sim=sim,
        derivative=hedged_derivative,
        hedging_derivatives=hedging_derivatives,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device),
        transaction_costs=transaction_costs
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
    
    metrics = {
        "episode": 0,
        "loss": float(torch.abs(terminal_errors).mean().item()),
        "reward": -float(torch.abs(terminal_errors).mean().item()),
        "trajectories": trajectories,
        "RL_positions": RL_positions,
        "S_traj": S_traj,
        "V_traj": V_traj,
        "O_traj": O_traj,
        "env": env
    }
    
    try:
        from src.visualization.plot_results import plot_episode_results
        plot_episode_results(episode=0, metrics=metrics, config=config)
        logging.info("Inference plots generated successfully")
    except Exception as e:
        logging.warning(f"Plot generation failed: {e}")


def train(
    config: Dict[str, Any],
    HedgingSim,
    hedged_derivative,
    hedging_derivatives,
    visualize: bool = True,
    initial_model: Optional[PolicyNetGARCH] = None
) -> PolicyNetGARCH:
    """Main training loop."""
    
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(config["training"]["device"])
    logging.info(f"Using device: {device}")
    
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
                device=device
            )
            
            if episode % checkpoint_freq == 0:
                save_checkpoint(policy_net, config, episode)
            
            if visualize and episode % plot_freq == 0:
                try:
                    from src.visualization.plot_results import plot_episode_results
                    plot_episode_results(episode, metrics, config)
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
        
        except Exception as e:
            logging.exception(f"Error during episode {episode}: {e}")
            raise
    
    final_path = config["output"]["model_save_path"]
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
    
    # ============================================================
    # AUTO-DETECT DEVICE: Use CUDA if available, otherwise CPU
    # This OVERRIDES whatever device is in the config file
    # ============================================================
    auto_device = auto_detect_device()
    config["training"]["device"] = auto_device
    config["precomputation"]["device"] = auto_device
    
    setup_logging(config)
    
    logging.info("=" * 70)
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
    
    # Check if hedged derivative needs precomputation
    needs_hedged_precompute = hedged_type in ["vanilla", "barrier"]
    
    if needs_hedged_precompute:
        hedged_maturity_days = int(config["hedged_option"]["T"] * 252)
        
        # Check if this maturity is already in the list (from hedging instruments)
        if hedged_maturity_days not in config["instruments"]["maturities"]:
            if not hasattr(precomputation_manager, 'maturities'):
                precomputation_manager.maturities = []
            if hedged_maturity_days not in precomputation_manager.maturities:
                precomputation_manager.maturities.append(hedged_maturity_days)
                logging.info(
                    f"{hedged_type.capitalize()} hedged derivative at maturity {hedged_maturity_days} days "
                    f"requires precomputation - added to list"
                )
    
    precomputed_data = precomputation_manager.precompute_all()
    logging.info(f"Precomputation complete for maturities: {list(precomputed_data.keys())}")
    
    # Verify all needed maturities are precomputed
    if needs_hedged_precompute:
        hedged_maturity_days = int(config["hedged_option"]["T"] * 252)
        if hedged_maturity_days not in precomputed_data:
            logging.warning(f"Maturity {hedged_maturity_days} not in precomputed data, computing now...")
            precomputation_manager.precompute_for_maturity(hedged_maturity_days)
            precomputed_data[hedged_maturity_days] = precomputation_manager.get_precomputed_data(hedged_maturity_days)
            logging.info(f"Precomputation complete for N={hedged_maturity_days}")
    
    hedged_derivative, hedging_derivatives = setup_derivatives_from_precomputed(
        config, precomputed_data
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
        initial_model=initial_model
    )
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
