"""
Main training script for GARCH-based option hedging with RL.

This script loads configuration from YAML, initializes all components,
and runs the training loop. It replaces the train_garch function with
a more modular, configuration-driven approach.

Usage:
    python train.py --config cfgs/config_vanilla_2inst.yaml
    python train.py --config cfgs/config_barrier_2inst.yaml
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

# Import your modules
from src.agents.policy_net_garch_flexible import PolicyNetGARCH, HedgingEnvGARCH
from src.option_greek.precompute import create_precomputation_manager_from_config
from src.visualization.plot_results import compute_rl_metrics
from derivative_factory import setup_derivatives_from_precomputed


logger = logging.getLogger(__name__)


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


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    n_inst = config["instruments"]["n_hedging_instruments"]
    
    if n_inst < 1 or n_inst > 4:
        raise ValueError(f"n_hedging_instruments must be 1-4, got {n_inst}")
    
    # If we have vanilla options as hedging instruments (n_inst > 1)
    if n_inst > 1:
        n_strikes = len(config["instruments"]["strikes"])
        n_types = len(config["instruments"]["types"])
        n_maturities = len(config["instruments"]["maturities"])
        
        # All should have length n_inst - 1 (excluding stock)
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
    
    # Validate hedged option config
    hedged_cfg = config["hedged_option"]
    deriv_type = hedged_cfg["type"].lower()
    
    if deriv_type not in ["vanilla", "barrier"]:
        raise ValueError(f"Invalid hedged_option type: {deriv_type}")
    
    if hedged_cfg["option_type"] not in valid_types:
        raise ValueError(
            f"Invalid hedged_option option_type: {hedged_cfg['option_type']}"
        )
    
    if hedged_cfg["side"] not in ["long", "short"]:
        raise ValueError(
            f"Invalid hedged_option side: {hedged_cfg['side']}"
        )
    
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


# In train.py, update train_episode function:

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
        TCP=config["simulation"]["TCP"],
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
        transaction_costs=config.get("transaction_costs", {
            'stock': config["simulation"]["TCP"],
            'vanilla_option': config["simulation"]["TCP"] * 10,
            'barrier_option': config["simulation"]["TCP"] * 20
        })
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
    n_inst = config["instruments"]["n_hedging_instruments"]
    checkpoint_path = config["output"]["checkpoint_path"].format(n_inst=n_inst)
    
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
    
    # Create simulation
    hedged_cfg = config["hedged_option"]
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
        TCP=config["simulation"]["TCP"],
        seed=config["training"]["seed"]
    )
    
    # Create environment
    env = HedgingEnvGARCH(
        sim=sim,
        derivative=hedged_derivative,
        hedging_derivatives=hedging_derivatives,
        garch_params=config["garch"],
        n_hedging_instruments=config["instruments"]["n_hedging_instruments"],
        dt_min=config["environment"]["dt_min"],
        device=str(device)
    )
    
    env.reset()
    
    # Run inference
    with torch.no_grad():
        S_traj, V_traj, O_traj, obs_sequence, RL_positions = \
            env.simulate_trajectory_and_get_observations(policy_net)
        
        terminal_errors, trajectories = env.simulate_full_trajectory(RL_positions, O_traj)
    
    # Compute metrics
    terminal_hedge_error_rl, rl_metrics = compute_rl_metrics(
        env, RL_positions, trajectories, O_traj
    )
    
    # Log metrics
    logging.info(
        f"Inference Results - MSE: {rl_metrics['mse']:.6f} | "
        f"SMSE: {rl_metrics['smse']:.6f} | CVaR95: {rl_metrics['cvar_95']:.6f}"
    )
    
    # Create metrics dict for visualization
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
    
    # Generate plots
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
    
    # Set seeds
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device = torch.device(config["training"]["device"])
    logging.info(f"Using device: {device}")
    
    # Create policy network and optimizer
    policy_net = create_policy_network(config, device)
    
    # Load initial model if provided
    if initial_model is not None:
        policy_net.load_state_dict(initial_model.state_dict())
        logging.info("Initialized policy network from pretrained model")
    
    optimizer = create_optimizer(policy_net, config)
    
    # Training loop
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
            
            # Save checkpoint
            if episode % checkpoint_freq == 0:
                save_checkpoint(policy_net, config, episode)
            
            # Visualization
            if visualize and episode % plot_freq == 0:
                try:
                    from src.visualization.plot_results import plot_episode_results
                    plot_episode_results(episode, metrics, config)
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
        
        except Exception as e:
            logging.exception(f"Error during episode {episode}: {e}")
            raise
    
    # Save final model
    n_inst = config["instruments"]["n_hedging_instruments"]
    final_path = config["output"]["model_save_path"].format(n_inst=n_inst)
    torch.save(policy_net.state_dict(), final_path)
    logging.info(f"Training finished. Model saved to {final_path}")
    
    return policy_net


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train GARCH-based option hedging with RL"
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
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Validate config
    validate_config(config)
    
    # Setup device
    device = torch.device(config["training"]["device"])
    logging.info(f"Using device: {device}")
    
    # Import HedgingSim
    try:
        from src.simulation.hedging_sim import HedgingSim
    except ImportError:
        logging.error("Could not import HedgingSim. Please adjust import path.")
        sys.exit(1)
    
    # FIXED PRECOMPUTATION LOGIC:
    # Always precompute for hedging instruments (they're always vanilla)
    # Also precompute for hedged derivative if it's vanilla
    hedged_type = config["hedged_option"]["type"].lower()
    
    logging.info("Starting precomputation...")
    logging.info(f"Hedged derivative type: {hedged_type}")
    
    # Precompute hedging instruments (always needed)
    precomputation_manager = create_precomputation_manager_from_config(config)
    precomputed_data = precomputation_manager.precompute_all()
    logging.info("Precomputation complete for hedging instruments")
    
    # If hedged derivative is vanilla, its coefficients are already in precomputed_data
    # If it's barrier/exotic, derivative_factory will handle Monte Carlo pricing
    
    # Setup derivatives (vanilla or barrier based on config)
    hedged_derivative, hedging_derivatives = setup_derivatives_from_precomputed(
        config, precomputed_data
    )
    
    # Inference-only mode
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
    
    # Training mode with optional model loading
    initial_model = None
    if args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        initial_model = create_policy_network(config, device)
        initial_model.load_state_dict(torch.load(args.load_model, map_location=device))
        logging.info("Model loaded - will continue training from checkpoint")
    
    # Train
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
