"""
Streamlit web interface for RL agent training and visualization.

This module provides an interactive web interface for training and evaluating
reinforcement learning agents.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import gymnasium as gym
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents import ActorCriticAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_environment(env_name: str) -> gym.Env:
    """Create and return a gymnasium environment."""
    return gym.make(env_name)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'losses': []
        }
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0


def create_agent(env_name: str, config: dict) -> ActorCriticAgent:
    """Create a new agent with the specified configuration."""
    env = create_environment(env_name)
    agent = ActorCriticAgent(
        env=env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        hidden_size=config['hidden_size'],
        activation=config['activation'],
        dropout=config['dropout']
    )
    return agent, env


def train_episode(agent: ActorCriticAgent) -> dict:
    """Train the agent for one episode."""
    return agent.train_episode()


def plot_rewards(rewards: list) -> go.Figure:
    """Create an interactive plot of episode rewards."""
    fig = go.Figure()
    
    # Moving average
    if len(rewards) > 10:
        window_size = min(50, len(rewards) // 10)
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        
        fig.add_trace(go.Scatter(
            y=rewards,
            mode='lines',
            name='Episode Rewards',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window_size})',
            line=dict(color='red', width=2)
        ))
    else:
        fig.add_trace(go.Scatter(
            y=rewards,
            mode='lines+markers',
            name='Episode Rewards',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title='Training Progress - Episode Rewards',
        xaxis_title='Episode',
        yaxis_title='Reward',
        hovermode='x unified'
    )
    
    return fig


def plot_losses(losses: list) -> go.Figure:
    """Create an interactive plot of training losses."""
    if not losses:
        return go.Figure()
    
    episodes = [l['episode'] for l in losses]
    actor_losses = [l['actor_loss'] for l in losses]
    critic_losses = [l['critic_loss'] for l in losses]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=episodes,
        y=actor_losses,
        mode='lines',
        name='Actor Loss',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes,
        y=critic_losses,
        mode='lines',
        name='Critic Loss',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Training Losses',
        xaxis_title='Episode',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RL Agent Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning Agent Training Dashboard")
    st.markdown("Train and visualize Actor-Critic agents with interactive controls")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_options = {
        "CartPole-v1": "Classic control task",
        "MountainCar-v0": "Sparse reward environment",
        "Acrobot-v1": "Underactuated pendulum",
        "LunarLander-v2": "Continuous control"
    }
    
    selected_env = st.sidebar.selectbox(
        "Environment",
        options=list(env_options.keys()),
        help="Select the training environment"
    )
    
    st.sidebar.markdown(f"**Description:** {env_options[selected_env]}")
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-1,
        value=1e-3,
        format="%.0e"
    )
    
    gamma = st.sidebar.slider(
        "Discount Factor (γ)",
        min_value=0.8,
        max_value=0.999,
        value=0.99,
        step=0.001
    )
    
    hidden_size = st.sidebar.slider(
        "Hidden Layer Size",
        min_value=32,
        max_value=512,
        value=128,
        step=32
    )
    
    activation = st.sidebar.selectbox(
        "Activation Function",
        options=["relu", "tanh", "elu"],
        index=0
    )
    
    dropout = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.05
    )
    
    # Training controls
    st.sidebar.subheader("Training Controls")
    
    episodes_to_train = st.sidebar.number_input(
        "Episodes to Train",
        min_value=1,
        max_value=1000,
        value=10,
        step=1
    )
    
    eval_frequency = st.sidebar.number_input(
        "Evaluation Frequency",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )
    
    # Create agent button
    if st.sidebar.button("🔄 Create New Agent", type="primary"):
        config = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'hidden_size': hidden_size,
            'activation': activation,
            'dropout': dropout
        }
        
        with st.spinner("Creating agent..."):
            agent, env = create_agent(selected_env, config)
            st.session_state.agent = agent
            st.session_state.env = env
            st.session_state.training_metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'eval_rewards': [],
                'losses': []
            }
        
        st.sidebar.success("Agent created successfully!")
    
    # Training section
    if st.session_state.agent is not None:
        st.header("Training")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Training", type="primary", disabled=st.session_state.is_training):
                st.session_state.is_training = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for episode in range(episodes_to_train):
                    # Train one episode
                    metrics = train_episode(st.session_state.agent)
                    
                    # Store metrics
                    st.session_state.training_metrics['episode_rewards'].append(metrics['episode_reward'])
                    st.session_state.training_metrics['episode_lengths'].append(metrics['episode_length'])
                    
                    if 'actor_loss' in metrics:
                        st.session_state.training_metrics['losses'].append({
                            'episode': episode,
                            'actor_loss': metrics['actor_loss'],
                            'critic_loss': metrics['critic_loss'],
                            'total_loss': metrics['total_loss']
                        })
                    
                    # Evaluation
                    if (episode + 1) % eval_frequency == 0:
                        eval_metrics = st.session_state.agent.evaluate(5)
                        st.session_state.training_metrics['eval_rewards'].append({
                            'episode': episode,
                            'mean_reward': eval_metrics['eval_mean_reward'],
                            'std_reward': eval_metrics['eval_std_reward']
                        })
                    
                    # Update progress
                    progress = (episode + 1) / episodes_to_train
                    progress_bar.progress(progress)
                    status_text.text(f"Episode {episode + 1}/{episodes_to_train} - Reward: {metrics['episode_reward']:.2f}")
                
                st.session_state.is_training = False
                st.success("Training completed!")
        
        # Metrics display
        if st.session_state.training_metrics['episode_rewards']:
            st.subheader("Training Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            rewards = st.session_state.training_metrics['episode_rewards']
            
            with col1:
                st.metric("Total Episodes", len(rewards))
            
            with col2:
                st.metric("Latest Reward", f"{rewards[-1]:.2f}")
            
            with col3:
                if len(rewards) >= 10:
                    recent_avg = np.mean(rewards[-10:])
                    st.metric("Recent Average (10)", f"{recent_avg:.2f}")
                else:
                    st.metric("Average", f"{np.mean(rewards):.2f}")
            
            with col4:
                st.metric("Best Reward", f"{np.max(rewards):.2f}")
            
            # Plots
            st.subheader("Training Progress")
            
            tab1, tab2 = st.tabs(["Rewards", "Losses"])
            
            with tab1:
                fig_rewards = plot_rewards(rewards)
                st.plotly_chart(fig_rewards, use_container_width=True)
            
            with tab2:
                fig_losses = plot_losses(st.session_state.training_metrics['losses'])
                if fig_losses.data:
                    st.plotly_chart(fig_losses, use_container_width=True)
                else:
                    st.info("No loss data available yet. Losses are computed during training.")
            
            # Evaluation results
            if st.session_state.training_metrics['eval_rewards']:
                st.subheader("Evaluation Results")
                
                eval_data = st.session_state.training_metrics['eval_rewards']
                eval_episodes = [d['episode'] for d in eval_data]
                eval_rewards = [d['mean_reward'] for d in eval_data]
                eval_stds = [d['std_reward'] for d in eval_data]
                
                fig_eval = go.Figure()
                fig_eval.add_trace(go.Scatter(
                    x=eval_episodes,
                    y=eval_rewards,
                    error_y=dict(type='data', array=eval_stds),
                    mode='lines+markers',
                    name='Evaluation Reward',
                    line=dict(color='green', width=2)
                ))
                
                fig_eval.update_layout(
                    title='Evaluation Performance',
                    xaxis_title='Episode',
                    yaxis_title='Mean Reward',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_eval, use_container_width=True)
        
        # Model management
        st.subheader("Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Model"):
                if st.session_state.agent:
                    model_path = f"models/{selected_env}_actor_critic.pth"
                    os.makedirs("models", exist_ok=True)
                    st.session_state.agent.save(model_path)
                    st.success(f"Model saved to {model_path}")
        
        with col2:
            uploaded_file = st.file_uploader("📁 Load Model", type=['pth'])
            if uploaded_file and st.button("Load"):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    st.session_state.agent.load(temp_path)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    else:
        st.info("👈 Please create an agent using the sidebar controls to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, PyTorch, and Gymnasium | "
        "Actor-Critic Implementation"
    )


if __name__ == "__main__":
    main()
