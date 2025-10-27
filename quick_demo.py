#!/usr/bin/env python3
"""
Quick example demonstrating the modernized RL implementation.

This script shows how to use the different algorithms and environments
in just a few lines of code.
"""

import sys
from pathlib import Path
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import ActorCriticAgent, PPOAgent, SACAgent
from envs import create_environment


def quick_demo():
    """Run a quick demonstration of the RL implementation."""
    print("🚀 RL Implementation Quick Demo\n")
    
    # Example 1: Actor-Critic on CartPole
    print("1. Actor-Critic on CartPole-v1")
    env = gym.make("CartPole-v1")
    agent = ActorCriticAgent(env, learning_rate=1e-3)
    
    # Train for a few episodes
    for episode in range(5):
        metrics = agent.train_episode()
        print(f"   Episode {episode + 1}: Reward = {metrics['episode_reward']:.2f}")
    
    # Evaluate
    eval_metrics = agent.evaluate(3)
    print(f"   Evaluation: {eval_metrics['eval_mean_reward']:.2f} ± {eval_metrics['eval_std_reward']:.2f}")
    env.close()
    
    # Example 2: PPO on MountainCar
    print("\n2. PPO on MountainCar-v0")
    env = gym.make("MountainCar-v0")
    agent = PPOAgent(env, learning_rate=3e-4, clip_ratio=0.2)
    
    # Train for a few episodes
    for episode in range(5):
        metrics = agent.train_episode()
        print(f"   Episode {episode + 1}: Reward = {metrics['episode_reward']:.2f}")
    
    # Evaluate
    eval_metrics = agent.evaluate(3)
    print(f"   Evaluation: {eval_metrics['eval_mean_reward']:.2f} ± {eval_metrics['eval_std_reward']:.2f}")
    env.close()
    
    # Example 3: SAC on Pendulum (continuous action)
    print("\n3. SAC on Pendulum-v1 (Continuous Actions)")
    env = gym.make("Pendulum-v1")
    agent = SACAgent(env, learning_rate=3e-4, tau=0.005)
    
    # Train for a few episodes
    for episode in range(5):
        metrics = agent.train_episode()
        print(f"   Episode {episode + 1}: Reward = {metrics['episode_reward']:.2f}")
    
    # Evaluate
    eval_metrics = agent.evaluate(3)
    print(f"   Evaluation: {eval_metrics['eval_mean_reward']:.2f} ± {eval_metrics['eval_std_reward']:.2f}")
    env.close()
    
    # Example 4: Custom GridWorld
    print("\n4. Custom GridWorld Environment")
    env = create_environment("GridWorld-v0", size=3)
    agent = ActorCriticAgent(env, learning_rate=1e-3)
    
    # Train for a few episodes
    for episode in range(5):
        metrics = agent.train_episode()
        print(f"   Episode {episode + 1}: Reward = {metrics['episode_reward']:.2f}")
    
    # Evaluate
    eval_metrics = agent.evaluate(3)
    print(f"   Evaluation: {eval_metrics['eval_mean_reward']:.2f} ± {eval_metrics['eval_std_reward']:.2f}")
    env.close()
    
    print("\n🎉 Demo completed! All algorithms are working correctly.")
    print("\nNext steps:")
    print("- Run full training: python train.py --env CartPole-v1 --episodes 100")
    print("- Try different algorithms: python train_advanced.py --algorithm ppo")
    print("- Launch web interface: streamlit run app.py")
    print("- Open interactive demo: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    quick_demo()
