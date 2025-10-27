"""
Environment utilities and wrappers.

This module provides utilities for working with different environments
and creating custom environment wrappers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces


class GridWorldEnv(Env):
    """Simple GridWorld environment for testing RL algorithms."""
    
    def __init__(self, size: int = 5, max_steps: int = 100):
        """
        Initialize GridWorld environment.
        
        Args:
            size: Size of the grid (size x size)
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_deltas = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1), # left
            3: (0, 1)    # right
        }
        
        # Initialize state
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.current_pos = self.start_pos
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_pos = self.start_pos
        self.current_step = 0
        
        return np.array(self.current_pos, dtype=np.int32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.current_step += 1
        
        # Get action delta
        delta = self.action_deltas[action]
        new_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])
        
        # Check bounds
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            self.current_pos = new_pos
        
        # Calculate reward
        if self.current_pos == self.goal_pos:
            reward = 10.0
            terminated = True
        else:
            reward = -0.1  # Small negative reward for each step
            terminated = False
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        return np.array(self.current_pos, dtype=np.int32), reward, terminated, truncated, {}
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if mode == "human":
            grid = np.zeros((self.size, self.size), dtype=str)
            grid.fill('.')
            grid[self.current_pos] = 'A'  # Agent
            grid[self.goal_pos] = 'G'     # Goal
            
            print("\n".join([" ".join(row) for row in grid]))
            print(f"Position: {self.current_pos}, Step: {self.current_step}")
        
        return None


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping."""
    
    def __init__(self, env: Env, reward_scale: float = 1.0, reward_shift: float = 0.0):
        """
        Initialize reward shaping wrapper.
        
        Args:
            env: Environment to wrap
            reward_scale: Scale factor for rewards
            reward_shift: Shift factor for rewards
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Take a step with reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward * self.reward_scale + self.reward_shift
        
        return obs, shaped_reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    """Wrapper for repeating actions."""
    
    def __init__(self, env: Env, repeat: int = 1):
        """
        Initialize action repeat wrapper.
        
        Args:
            env: Environment to wrap
            repeat: Number of times to repeat each action
        """
        super().__init__(env)
        self.repeat = repeat
        self.current_repeat = 0
        self.last_action = None
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Take a step with action repetition."""
        if self.current_repeat == 0:
            self.last_action = action
        
        obs, reward, terminated, truncated, info = self.env.step(self.last_action)
        
        self.current_repeat += 1
        if self.current_repeat >= self.repeat:
            self.current_repeat = 0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        self.current_repeat = 0
        self.last_action = None
        return self.env.reset(**kwargs)


def create_environment(
    env_name: str, 
    render_mode: Optional[str] = None,
    **kwargs: Any
) -> Env:
    """
    Create an environment with optional wrappers.
    
    Args:
        env_name: Name of the environment
        render_mode: Render mode for the environment
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Created environment
    """
    if env_name == "GridWorld-v0":
        env = GridWorldEnv(**kwargs)
    else:
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
    
    return env


def get_environment_info(env: Env) -> Dict[str, Any]:
    """
    Get information about an environment.
    
    Args:
        env: Environment to analyze
        
    Returns:
        Dictionary containing environment information
    """
    info = {
        'name': env.spec.id if hasattr(env, 'spec') and env.spec else 'Unknown',
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space),
        'reward_range': env.reward_range if hasattr(env, 'reward_range') else 'Unknown',
        'max_episode_steps': getattr(env.spec, 'max_episode_steps', None) if hasattr(env, 'spec') and env.spec else None
    }
    
    return info
