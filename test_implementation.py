#!/usr/bin/env python3
"""
Test script to verify the RL implementation works correctly.

This script runs basic tests to ensure all components are functioning properly.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import ActorCriticAgent, PPOAgent, SACAgent
from envs import create_environment, GridWorldEnv


def test_actor_critic():
    """Test Actor-Critic agent."""
    print("Testing Actor-Critic agent...")
    
    env = gym.make("CartPole-v1")
    agent = ActorCriticAgent(env, learning_rate=1e-3, gamma=0.99)
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, training=True)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n
    
    # Test training episode
    metrics = agent.train_episode()
    assert 'episode_reward' in metrics
    assert 'episode_length' in metrics
    
    # Test evaluation
    eval_metrics = agent.evaluate(3)
    assert 'eval_mean_reward' in eval_metrics
    
    env.close()
    print("✓ Actor-Critic test passed")


def test_ppo():
    """Test PPO agent."""
    print("Testing PPO agent...")
    
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env, learning_rate=3e-4, gamma=0.99)
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, training=True)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n
    
    # Test training episode
    metrics = agent.train_episode()
    assert 'episode_reward' in metrics
    assert 'episode_length' in metrics
    
    # Test evaluation
    eval_metrics = agent.evaluate(3)
    assert 'eval_mean_reward' in eval_metrics
    
    env.close()
    print("✓ PPO test passed")


def test_sac():
    """Test SAC agent."""
    print("Testing SAC agent...")
    
    env = gym.make("Pendulum-v1")
    agent = SACAgent(env, learning_rate=3e-4, gamma=0.99)
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, training=True)
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape
    
    # Test training episode
    metrics = agent.train_episode()
    assert 'episode_reward' in metrics
    assert 'episode_length' in metrics
    
    # Test evaluation
    eval_metrics = agent.evaluate(3)
    assert 'eval_mean_reward' in eval_metrics
    
    env.close()
    print("✓ SAC test passed")


def test_gridworld():
    """Test GridWorld environment."""
    print("Testing GridWorld environment...")
    
    env = GridWorldEnv(size=5)
    
    # Test reset
    state, info = env.reset()
    assert state.shape == (2,)
    assert isinstance(state, np.ndarray)
    
    # Test step
    action = 0  # up
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    env.close()
    print("✓ GridWorld test passed")


def test_environment_creation():
    """Test environment creation utility."""
    print("Testing environment creation...")
    
    # Test standard environment
    env = create_environment("CartPole-v1")
    assert isinstance(env, gym.Env)
    env.close()
    
    # Test GridWorld
    env = create_environment("GridWorld-v0", size=3)
    assert isinstance(env, GridWorldEnv)
    env.close()
    
    print("✓ Environment creation test passed")


def test_model_saving_loading():
    """Test model saving and loading."""
    print("Testing model saving and loading...")
    
    env = gym.make("CartPole-v1")
    agent = ActorCriticAgent(env)
    
    # Train a bit
    for _ in range(5):
        agent.train_episode()
    
    # Save model
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model_path = f.name
    
    agent.save(model_path)
    
    # Load model
    new_agent = ActorCriticAgent(env)
    new_agent.load(model_path)
    
    # Verify loaded model
    assert len(new_agent.episode_rewards) == len(agent.episode_rewards)
    assert new_agent.training_step == agent.training_step
    
    # Clean up
    import os
    os.unlink(model_path)
    env.close()
    
    print("✓ Model saving/loading test passed")


def main():
    """Run all tests."""
    print("Running RL implementation tests...\n")
    
    try:
        test_actor_critic()
        test_ppo()
        test_sac()
        test_gridworld()
        test_environment_creation()
        test_model_saving_loading()
        
        print("\n🎉 All tests passed! The implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
