"""
Proximal Policy Optimization (PPO) implementation.

This module implements the PPO algorithm with clipping and multiple epochs
per update for improved sample efficiency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env

from .base_agent import BaseAgent


class PPONetwork(nn.Module):
    """PPO neural network with shared base layers."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the PPO network.
        
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
            Tuple of (action_logits, state_value)
        """
        base_features = self.base(state)
        
        # Actor output (action logits)
        action_logits = self.actor(base_features)
        
        # Critic output (state value)
        state_value = self.critic(base_features)
        
        return action_logits, state_value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for given state.
        
        Args:
            state: Input state tensor
            action: Optional action tensor (if None, sample from policy)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, state_value = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, state_value.squeeze()


class PPOAgent(BaseAgent):
    """PPO agent implementation."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        hidden_size: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "auto",
        **kwargs: Any
    ) -> None:
        """
        Initialize the PPO agent.
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            hidden_size: Size of hidden layers in the network
            activation: Activation function for the network
            dropout: Dropout probability
            device: Device to run computations on
            **kwargs: Additional arguments
        """
        super().__init__(env, learning_rate, gamma, device, **kwargs)
        
        # PPO-specific parameters
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize network
        self.network = PPONetwork(
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
        
        # Buffer for batch processing
        self.buffer_size = 0
    
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
            action_logits, value = self.network(state_tensor)
            
            if training:
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Store for training
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob)
                self.values.append(value.squeeze())
                
                return action.item()
            else:
                # Greedy action selection for evaluation
                return action_logits.argmax().item()
    
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
        self.buffer_size += 1
    
    def _should_update(self) -> bool:
        """Check if the agent should update (when buffer is full)."""
        return self.buffer_size >= self.batch_size
    
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
    
    def _compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's parameters using PPO.
        
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
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        old_values = torch.stack(self.values).to(self.device)
        dones = np.array(self.dones)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, old_values.cpu().numpy(), dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Get current policy
            action_logits, current_values = self.network(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            # Compute new log probabilities
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Compute policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(current_values, returns)
            
            # Compute entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
        
        # Clear stored data
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.buffer_size = 0
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy_loss': total_entropy_loss / self.ppo_epochs,
            'total_loss': total_loss / self.ppo_epochs,
            'mean_return': returns.mean().item(),
            'mean_advantage': advantages.mean().item()
        }
    
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
