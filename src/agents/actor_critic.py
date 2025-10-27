"""
Actor-Critic implementation with modern PyTorch practices.

This module implements the Actor-Critic algorithm with proper type hints,
error handling, and modern PyTorch features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env

from .base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network with shared base layers."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the Actor-Critic network.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            activation: Activation function ("relu", "tanh", "elu")
            dropout: Dropout probability
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared base layers
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        base_features = self.base(state)
        
        # Actor output (action probabilities)
        action_logits = self.actor(base_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output (state value)
        state_value = self.critic(base_features)
        
        return action_probs, state_value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for given state.
        
        Args:
            state: Input state tensor
            action: Optional action tensor (if None, sample from policy)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_probs, state_value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value.squeeze()


class ActorCriticAgent(BaseAgent):
    """Actor-Critic agent implementation."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "auto",
        **kwargs: Any
    ) -> None:
        """
        Initialize the Actor-Critic agent.
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            hidden_size: Size of hidden layers in the network
            activation: Activation function for the network
            dropout: Dropout probability
            device: Device to run computations on
            **kwargs: Additional arguments
        """
        super().__init__(env, learning_rate, gamma, device, **kwargs)
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize network
        self.network = ActorCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training data storage
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs, _ = self.network(state_tensor)
            
            if training:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.network.critic(self.network.base(state_tensor))
                
                # Store for training
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob)
                self.values.append(value.squeeze())
                
                return action.item()
            else:
                # Greedy action selection for evaluation
                return action_probs.argmax().item()
    
    def _store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience for training."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def _should_update(self) -> bool:
        """Check if the agent should update (after each episode)."""
        return len(self.states) > 0
    
    def _get_batch(self) -> Dict[str, Any]:
        """Get a batch of experience for training."""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'log_probs': self.log_probs,
            'values': self.values,
            'dones': self.dones
        }
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's parameters using stored experience.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        if not self.states:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = np.array(self.rewards)
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).to(self.device)
        dones = np.array(self.dones)
        
        # Compute returns
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + critic_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clear stored data
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'mean_return': returns.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns.
        
        Args:
            rewards: Array of rewards
            dones: Array of done flags
            
        Returns:
            Array of discounted returns
        """
        returns = np.zeros_like(rewards)
        G = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                G = 0
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
    def save(self, filepath: str) -> None:
        """Save the agent's state to a file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
