#!/usr/bin/env python3
"""
Advanced training script for multiple RL algorithms.

This script provides a unified interface for training Actor-Critic, PPO, and SAC agents
with various environments and configurations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import ActorCriticAgent, PPOAgent, SACAgent
from envs import create_environment, get_environment_info


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with rich formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(
    algorithm: str,
    env: gym.Env,
    config: Dict[str, Any]
) -> Any:
    """
    Create an agent based on the specified algorithm.
    
    Args:
        algorithm: Algorithm name ("actor_critic", "ppo", "sac")
        env: Environment to train on
        config: Configuration dictionary
        
    Returns:
        Created agent
    """
    agent_config = config.get('algorithms', {}).get(algorithm, {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    base_params = {
        'env': env,
        'learning_rate': training_config.get('learning_rate', 1e-3),
        'gamma': training_config.get('gamma', 0.99),
        'hidden_size': model_config.get('hidden_size', 128),
        'activation': model_config.get('activation', 'relu'),
        'dropout': model_config.get('dropout', 0.0)
    }
    
    if algorithm == "actor_critic":
        return ActorCriticAgent(**base_params)
    elif algorithm == "ppo":
        ppo_params = {
            **base_params,
            'clip_ratio': agent_config.get('clip_ratio', 0.2),
            'value_coef': agent_config.get('value_coef', 0.5),
            'entropy_coef': agent_config.get('entropy_coef', 0.01),
            'ppo_epochs': agent_config.get('ppo_epochs', 4),
            'batch_size': agent_config.get('batch_size', 64)
        }
        return PPOAgent(**ppo_params)
    elif algorithm == "sac":
        sac_params = {
            **base_params,
            'tau': agent_config.get('tau', 0.005),
            'alpha': agent_config.get('alpha', 0.2),
            'automatic_entropy_tuning': agent_config.get('automatic_entropy_tuning', True)
        }
        return SACAgent(**sac_params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(
    agent: Any,
    config: Dict[str, Any],
    console: Console,
    algorithm: str
) -> Dict[str, Any]:
    """
    Train the agent for the specified number of episodes.
    
    Args:
        agent: The agent to train
        config: Training configuration
        console: Rich console for output
        algorithm: Algorithm name
        
    Returns:
        Dictionary containing training results
    """
    episodes = config['training']['episodes']
    eval_frequency = config['evaluation']['eval_frequency']
    eval_episodes = config['evaluation']['eval_episodes']
    
    # Training metrics
    training_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'losses': []
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training {algorithm.upper()}...", total=episodes)
        
        for episode in range(episodes):
            # Train one episode
            metrics = agent.train_episode()
            
            # Store metrics
            training_metrics['episode_rewards'].append(metrics['episode_reward'])
            training_metrics['episode_lengths'].append(metrics['episode_length'])
            
            # Store losses based on algorithm
            if algorithm == "actor_critic" and 'actor_loss' in metrics:
                training_metrics['losses'].append({
                    'episode': episode,
                    'actor_loss': metrics['actor_loss'],
                    'critic_loss': metrics['critic_loss'],
                    'total_loss': metrics['total_loss']
                })
            elif algorithm == "ppo" and 'policy_loss' in metrics:
                training_metrics['losses'].append({
                    'episode': episode,
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss'],
                    'entropy_loss': metrics['entropy_loss'],
                    'total_loss': metrics['total_loss']
                })
            elif algorithm == "sac" and 'actor_loss' in metrics:
                training_metrics['losses'].append({
                    'episode': episode,
                    'actor_loss': metrics['actor_loss'],
                    'critic_loss': metrics['critic_loss'],
                    'alpha_loss': metrics['alpha_loss'],
                    'total_loss': metrics['actor_loss'] + metrics['critic_loss']
                })
            
            # Evaluation
            if (episode + 1) % eval_frequency == 0:
                eval_metrics = agent.evaluate(eval_episodes)
                training_metrics['eval_rewards'].append({
                    'episode': episode,
                    'mean_reward': eval_metrics['eval_mean_reward'],
                    'std_reward': eval_metrics['eval_std_reward']
                })
                
                console.print(
                    f"Episode {episode + 1}: "
                    f"Reward={metrics['episode_reward']:.2f}, "
                    f"Eval Reward={eval_metrics['eval_mean_reward']:.2f}±{eval_metrics['eval_std_reward']:.2f}"
                )
            
            progress.update(task, advance=1)
    
    return training_metrics


def plot_training_results(
    training_metrics: Dict[str, Any],
    algorithm: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training results including rewards and losses.
    
    Args:
        training_metrics: Dictionary containing training metrics
        algorithm: Algorithm name
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(training_metrics['episode_rewards'])
    axes[0, 0].set_title(f'{algorithm.upper()} - Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(training_metrics['episode_lengths'])
    axes[0, 1].set_title(f'{algorithm.upper()} - Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True)
    
    # Evaluation rewards
    if training_metrics['eval_rewards']:
        eval_episodes = [m['episode'] for m in training_metrics['eval_rewards']]
        eval_rewards = [m['mean_reward'] for m in training_metrics['eval_rewards']]
        eval_stds = [m['std_reward'] for m in training_metrics['eval_rewards']]
        
        axes[1, 0].errorbar(eval_episodes, eval_rewards, yerr=eval_stds, capsize=5)
        axes[1, 0].set_title(f'{algorithm.upper()} - Evaluation Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].grid(True)
    
    # Training losses
    if training_metrics['losses']:
        loss_episodes = [l['episode'] for l in training_metrics['losses']]
        
        if algorithm == "actor_critic":
            actor_losses = [l['actor_loss'] for l in training_metrics['losses']]
            critic_losses = [l['critic_loss'] for l in training_metrics['losses']]
            
            axes[1, 1].plot(loss_episodes, actor_losses, label='Actor Loss')
            axes[1, 1].plot(loss_episodes, critic_losses, label='Critic Loss')
        elif algorithm == "ppo":
            policy_losses = [l['policy_loss'] for l in training_metrics['losses']]
            value_losses = [l['value_loss'] for l in training_metrics['losses']]
            entropy_losses = [l['entropy_loss'] for l in training_metrics['losses']]
            
            axes[1, 1].plot(loss_episodes, policy_losses, label='Policy Loss')
            axes[1, 1].plot(loss_episodes, value_losses, label='Value Loss')
            axes[1, 1].plot(loss_episodes, entropy_losses, label='Entropy Loss')
        elif algorithm == "sac":
            actor_losses = [l['actor_loss'] for l in training_metrics['losses']]
            critic_losses = [l['critic_loss'] for l in training_metrics['losses']]
            alpha_losses = [l['alpha_loss'] for l in training_metrics['losses']]
            
            axes[1, 1].plot(loss_episodes, actor_losses, label='Actor Loss')
            axes[1, 1].plot(loss_episodes, critic_losses, label='Critic Loss')
            axes[1, 1].plot(loss_episodes, alpha_losses, label='Alpha Loss')
        
        axes[1, 1].set_title(f'{algorithm.upper()} - Training Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_training_summary(
    training_metrics: Dict[str, Any],
    algorithm: str,
    console: Console
) -> None:
    """Print a summary of training results."""
    rewards = training_metrics['episode_rewards']
    
    table = Table(title=f"{algorithm.upper()} Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Episodes", str(len(rewards)))
    table.add_row("Final Reward", f"{rewards[-1]:.2f}")
    table.add_row("Mean Reward (Last 100)", f"{np.mean(rewards[-100:]):.2f}")
    table.add_row("Max Reward", f"{np.max(rewards):.2f}")
    table.add_row("Min Reward", f"{np.min(rewards):.2f}")
    
    if training_metrics['eval_rewards']:
        eval_rewards = [m['mean_reward'] for m in training_metrics['eval_rewards']]
        table.add_row("Final Eval Reward", f"{eval_rewards[-1]:.2f}")
        table.add_row("Max Eval Reward", f"{np.max(eval_rewards):.2f}")
    
    console.print(table)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["actor_critic", "ppo", "sac"],
        default="actor_critic",
        help="Algorithm to use"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        help="Environment name (overrides config)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        help="Number of training episodes (overrides config)"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render environment during training"
    )
    parser.add_argument(
        "--save-model", 
        type=str,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--load-model", 
        type=str,
        help="Path to load pre-trained model"
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Show training plots"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    console = Console()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config['environment']['name'] = args.env
    if args.episodes:
        config['training']['episodes'] = args.episodes
    
    # Create environment
    render_mode = "human" if args.render else None
    env = create_environment(
        config['environment']['name'], 
        render_mode=render_mode
    )
    
    # Print environment info
    env_info = get_environment_info(env)
    console.print(f"[bold green]Environment:[/bold green] {env_info['name']}")
    console.print(f"[bold green]Algorithm:[/bold green] {args.algorithm.upper()}")
    console.print(f"[bold green]Episodes:[/bold green] {config['training']['episodes']}")
    console.print(f"[bold green]Observation Space:[/bold green] {env_info['observation_space']}")
    console.print(f"[bold green]Action Space:[/bold green] {env_info['action_space']}")
    
    # Create agent
    agent = create_agent(args.algorithm, env, config)
    
    # Load pre-trained model if specified
    if args.load_model:
        agent.load(args.load_model)
        console.print(f"[bold green]Loaded model from:[/bold green] {args.load_model}")
    
    # Train agent
    console.print(f"[bold blue]Starting {args.algorithm.upper()} training...[/bold blue]")
    training_metrics = train_agent(agent, config, console, args.algorithm)
    
    # Print summary
    print_training_summary(training_metrics, args.algorithm, console)
    
    # Save model if specified
    if args.save_model:
        agent.save(args.save_model)
        console.print(f"[bold green]Model saved to:[/bold green] {args.save_model}")
    
    # Show plots if requested
    if args.plot:
        plot_training_results(training_metrics, args.algorithm)
    
    # Final evaluation
    console.print("[bold blue]Running final evaluation...[/bold blue]")
    final_eval = agent.evaluate(100)
    console.print(f"Final evaluation: {final_eval['eval_mean_reward']:.2f} ± {final_eval['eval_std_reward']:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
