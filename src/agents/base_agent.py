"""
Base agent class for reinforcement learning algorithms.

This module provides the foundation for implementing various RL agents
with common interfaces and utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env


class BaseAgent(ABC):
    """Base class for all reinforcement learning agents."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "auto",
        **kwargs: Any
    ) -> None:
        """
        Initialize the base agent.
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            device: Device to run computations on ("cpu", "cuda", or "auto")
            **kwargs: Additional arguments passed to subclasses
        """
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_step = 0
        
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's parameters using a batch of experience.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's state to a file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's state from a file."""
        pass
    
    def train_episode(self) -> Dict[str, float]:
        """
        Run a single training episode.
        
        Returns:
            Dictionary containing episode metrics
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience (to be implemented by subclasses)
            self._store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update agent if needed
        if hasattr(self, '_should_update') and self._should_update():
            metrics = self.update(self._get_batch())
        else:
            metrics = {}
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.training_step += 1
        
        metrics.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_episodes': len(self.episode_rewards)
        })
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_mean_length': np.mean(eval_lengths),
            'eval_std_length': np.std(eval_lengths)
        }
    
    def _store_experience(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer (to be implemented by subclasses)."""
        pass
    
    def _should_update(self) -> bool:
        """Check if the agent should update (to be implemented by subclasses)."""
        return False
    
    def _get_batch(self) -> Dict[str, Any]:
        """Get a batch of experience for training (to be implemented by subclasses)."""
        return {}
