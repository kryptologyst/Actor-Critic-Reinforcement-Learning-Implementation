"""
Unit tests for the Actor-Critic implementation.

This module contains tests for the core components of the RL system.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents import ActorCriticAgent, ActorCriticNetwork


class TestActorCriticNetwork:
    """Test cases for the ActorCriticNetwork class."""
    
    def test_network_initialization(self):
        """Test network initialization with different parameters."""
        obs_dim = 4
        action_dim = 2
        
        network = ActorCriticNetwork(obs_dim, action_dim)
        
        assert network.obs_dim == obs_dim
        assert network.action_dim == action_dim
        assert network.hidden_size == 128  # default value
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        obs_dim = 4
        action_dim = 2
        batch_size = 5
        
        network = ActorCriticNetwork(obs_dim, action_dim)
        state = torch.randn(batch_size, obs_dim)
        
        action_probs, state_value = network(state)
        
        assert action_probs.shape == (batch_size, action_dim)
        assert state_value.shape == (batch_size, 1)
        
        # Check that action probabilities sum to 1
        assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    
    def test_get_action_and_value(self):
        """Test action sampling and value computation."""
        obs_dim = 4
        action_dim = 2
        
        network = ActorCriticNetwork(obs_dim, action_dim)
        state = torch.randn(1, obs_dim)
        
        action, log_prob, value = network.get_action_and_value(state)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        assert 0 <= action.item() < action_dim


class TestActorCriticAgent:
    """Test cases for the ActorCriticAgent class."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment."""
        return gym.make("CartPole-v1")
    
    @pytest.fixture
    def agent(self, env):
        """Create a test agent."""
        return ActorCriticAgent(env, learning_rate=1e-3, gamma=0.99)
    
    def test_agent_initialization(self, env):
        """Test agent initialization."""
        agent = ActorCriticAgent(env)
        
        assert agent.env == env
        assert agent.learning_rate == 1e-3
        assert agent.gamma == 0.99
        assert agent.obs_dim == 4  # CartPole observation space
        assert agent.action_dim == 2  # CartPole action space
    
    def test_action_selection_training(self, agent):
        """Test action selection in training mode."""
        state = np.random.randn(4)
        
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        
        # Check that experience was stored
        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.values) == 1
    
    def test_action_selection_evaluation(self, agent):
        """Test action selection in evaluation mode."""
        state = np.random.randn(4)
        
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        
        # Check that no experience was stored in evaluation mode
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
    
    def test_experience_storage(self, agent):
        """Test experience storage."""
        state = np.random.randn(4)
        action = 1
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent._store_experience(state, action, reward, next_state, done)
        
        assert len(agent.rewards) == 1
        assert len(agent.dones) == 1
        assert agent.rewards[0] == reward
        assert agent.dones[0] == done
    
    def test_returns_computation(self, agent):
        """Test discounted returns computation."""
        rewards = [1.0, 1.0, 1.0, 0.0]
        dones = [False, False, False, True]
        
        returns = agent._compute_returns(np.array(rewards), np.array(dones))
        
        expected_returns = np.array([
            1.0 + 0.99 * (1.0 + 0.99 * (1.0 + 0.99 * 0.0)),
            1.0 + 0.99 * (1.0 + 0.99 * 0.0),
            1.0 + 0.99 * 0.0,
            0.0
        ])
        
        assert np.allclose(returns, expected_returns, atol=1e-6)
    
    def test_update_with_empty_batch(self, agent):
        """Test update with empty batch."""
        metrics = agent.update({})
        
        assert metrics == {}
    
    def test_save_load(self, agent, tmp_path):
        """Test model saving and loading."""
        # Add some training data
        agent.training_step = 10
        agent.episode_rewards = [100.0, 200.0]
        agent.episode_lengths = [50, 60]
        
        # Save model
        model_path = tmp_path / "test_model.pth"
        agent.save(str(model_path))
        
        # Create new agent and load
        new_agent = ActorCriticAgent(agent.env)
        new_agent.load(str(model_path))
        
        assert new_agent.training_step == 10
        assert new_agent.episode_rewards == [100.0, 200.0]
        assert new_agent.episode_lengths == [50, 60]


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_training_episode(self):
        """Test complete training episode."""
        env = gym.make("CartPole-v1")
        agent = ActorCriticAgent(env)
        
        # Run one training episode
        metrics = agent.train_episode()
        
        assert 'episode_reward' in metrics
        assert 'episode_length' in metrics
        assert 'total_episodes' in metrics
        assert metrics['total_episodes'] == 1
        
        env.close()
    
    def test_evaluation(self):
        """Test agent evaluation."""
        env = gym.make("CartPole-v1")
        agent = ActorCriticAgent(env)
        
        # Run evaluation
        eval_metrics = agent.evaluate(num_episodes=3)
        
        assert 'eval_mean_reward' in eval_metrics
        assert 'eval_std_reward' in eval_metrics
        assert 'eval_mean_length' in eval_metrics
        assert 'eval_std_length' in eval_metrics
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
