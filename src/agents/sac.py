"""
Soft Actor-Critic (SAC) implementation for continuous action spaces.

This module implements the SAC algorithm with automatic entropy tuning
and twin Q-networks for improved stability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env

from .base_agent import BaseAgent


class SACNetwork(nn.Module):
    """SAC neural network with actor and critic components."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        activation: str = "relu",
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the SAC network.
        
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
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim * 2)  # mean and log_std
        )
        
        # Critic networks (twin Q-networks)
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward_actor(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        output = self.actor(state)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp log_std for stability
        return mean, log_std
    
    def forward_critic(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through critic networks.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Tuple of (q1, q2)
        """
        state_action = torch.cat([state, action], dim=-1)
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        return q1, q2
    
    def sample_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            state: Input state tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward_actor(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Compute log probability with tanh transformation
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return action, log_prob
        
        return action, torch.zeros_like(action)


class SACAgent(BaseAgent):
    """SAC agent implementation for continuous action spaces."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        hidden_size: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "auto",
        **kwargs: Any
    ) -> None:
        """
        Initialize the SAC agent.
        
        Args:
            env: The environment to interact with
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target networks
            alpha: Entropy regularization coefficient
            automatic_entropy_tuning: Whether to automatically tune entropy coefficient
            hidden_size: Size of hidden layers in the network
            activation: Activation function for the network
            dropout: Dropout probability
            device: Device to run computations on
            **kwargs: Additional arguments
        """
        super().__init__(env, learning_rate, gamma, device, **kwargs)
        
        # SAC-specific parameters
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_range = [
            env.action_space.low[0],
            env.action_space.high[0]
        ]
        
        # Initialize networks
        self.network = SACNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout
        ).to(self.device)
        
        # Target networks
        self.target_network = SACNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.network.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = optim.Adam(
            list(self.network.critic1.parameters()) + 
            list(self.network.critic2.parameters()), 
            lr=learning_rate
        )
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 100000
        self.batch_size = 256
        
        # Training data storage
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.network.sample_action(state_tensor, deterministic=not training)
            action = action.cpu().numpy()[0]
            
            # Scale action to environment range
            action = self.action_range[0] + (action + 1) / 2 * (self.action_range[1] - self.action_range[0])
            
            return action
    
    def _store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        # Normalize action to [-1, 1] range
        normalized_action = 2 * (action - self.action_range[0]) / (self.action_range[1] - self.action_range[0]) - 1
        
        self.replay_buffer.append({
            'state': state,
            'action': normalized_action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _should_update(self) -> bool:
        """Check if the agent should update (when buffer has enough samples)."""
        return len(self.replay_buffer) >= self.batch_size
    
    def _get_batch(self) -> Dict[str, Any]:
        """Get a batch of experience for training."""
        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
        
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in batch]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's parameters using SAC.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update critic networks
        with torch.no_grad():
            next_actions, next_log_probs = self.network.sample_action(next_states)
            target_q1, target_q2 = self.target_network.forward_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q * (~dones).unsqueeze(-1)
        
        current_q1, current_q2 = self.network.forward_critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor network
        new_actions, log_probs = self.network.sample_action(states)
        q1_new, q2_new = self.network.forward_critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy coefficient
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else 0,
            'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha,
            'mean_q': q_new.mean().item()
        }
    
    def save(self, filepath: str) -> None:
        """Save the agent's state to a file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
