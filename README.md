# Actor-Critic Reinforcement Learning Implementation

A comprehensive implementation of the Actor-Critic algorithm with support for multiple environments, advanced features, and interactive interfaces.

## Features

- **Modern PyTorch Implementation**: Clean, well-documented code with type hints and error handling
- **Multiple Environments**: Support for CartPole, MountainCar, Acrobot, and LunarLander
- **Interactive Interfaces**: Both CLI and Streamlit web interface
- **Advanced Visualization**: Real-time training plots and performance metrics
- **Model Management**: Save/load trained models with checkpointing
- **Configuration System**: YAML-based configuration for easy experimentation
- **Extensible Architecture**: Base classes for implementing additional RL algorithms

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Actor-Critic-Reinforcement-Learning-Implementation.git
cd Actor-Critic-Reinforcement-Learning-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

Train an Actor-Critic agent on CartPole:

```bash
python train.py --env CartPole-v1 --episodes 500 --plot
```

With custom configuration:

```bash
python train.py --config config/config.yaml --episodes 1000 --render
```

#### Web Interface

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` for an interactive training experience.

## 📁 Project Structure

```
├── src/
│   └── agents/
│       ├── __init__.py
│       ├── base_agent.py          # Base agent class
│       └── actor_critic.py        # Actor-Critic implementation
├── config/
│   └── config.yaml               # Configuration file
├── notebooks/                    # Jupyter notebooks for experimentation
├── tests/                        # Unit tests
├── logs/                         # Training logs
├── models/                       # Saved models
├── plots/                        # Generated plots
├── train.py                      # CLI training script
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Supported Environments

| Environment | Description | Action Space | Observation Space |
|-------------|-------------|--------------|-------------------|
| CartPole-v1 | Balance a pole on a cart | Discrete (2) | Continuous (4) |
| MountainCar-v0 | Drive up a mountain | Discrete (3) | Continuous (2) |
| Acrobot-v1 | Swing up an underactuated pendulum | Discrete (3) | Continuous (6) |
| LunarLander-v2 | Land a spacecraft | Discrete (4) | Continuous (8) |

## Configuration

The project uses YAML configuration files for easy parameter tuning. Key configuration sections:

### Environment Settings
```yaml
environment:
  name: "CartPole-v1"
  max_episode_steps: 500
  render_mode: null
```

### Training Parameters
```yaml
training:
  episodes: 1000
  gamma: 0.99
  learning_rate: 1e-3
  batch_size: 64
```

### Model Architecture
```yaml
model:
  hidden_size: 128
  activation: "relu"
  dropout: 0.0
```

## Algorithm Details

### Actor-Critic Method

The Actor-Critic algorithm combines the benefits of policy-based (Actor) and value-based (Critic) methods:

- **Actor**: Learns the policy π(a|s) to select actions
- **Critic**: Learns the value function V(s) to evaluate states
- **Advantage**: Uses A(s,a) = Q(s,a) - V(s) to reduce variance in policy updates

### Network Architecture

```
Input State → Shared Base Layers → Actor Head (Policy)
                              → Critic Head (Value)
```

- **Shared Base**: Common feature extraction layers
- **Actor Head**: Outputs action probabilities
- **Critic Head**: Outputs state values

## Training and Evaluation

### Training Metrics

- **Episode Rewards**: Total reward per episode
- **Episode Lengths**: Number of steps per episode
- **Actor Loss**: Policy gradient loss
- **Critic Loss**: Value function MSE loss
- **Advantages**: Normalized advantage estimates

### Evaluation

The agent is evaluated periodically during training using greedy action selection:

```python
eval_metrics = agent.evaluate(num_episodes=10)
print(f"Mean reward: {eval_metrics['eval_mean_reward']:.2f}")
```

## Visualization Features

### Real-time Plots

- **Training Progress**: Episode rewards with moving averages
- **Loss Curves**: Actor and critic loss over time
- **Evaluation Results**: Performance on held-out episodes
- **Learning Curves**: Convergence analysis

### Interactive Dashboard

The Streamlit interface provides:
- Real-time parameter adjustment
- Live training progress
- Interactive plots
- Model management tools

## Model Management

### Saving Models

```python
agent.save("models/my_agent.pth")
```

### Loading Models

```python
agent.load("models/my_agent.pth")
```

### Checkpointing

Models are automatically saved during training based on configuration:

```yaml
checkpointing:
  save_frequency: 100
  max_checkpoints: 5
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Development

### Code Style

The project follows PEP 8 style guidelines. Format code with:

```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

Run type checking with mypy:

```bash
mypy src/
```

## Performance Tips

1. **Environment Selection**: Start with CartPole-v1 for quick experimentation
2. **Learning Rate**: Use 1e-3 to 1e-4 for stable learning
3. **Network Size**: 128-256 hidden units work well for most tasks
4. **Evaluation**: Evaluate every 50-100 episodes for monitoring
5. **Patience**: Some environments (MountainCar) require many episodes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: Check if GPU is being utilized
3. **Poor Performance**: Try different learning rates or network sizes
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:

```bash
python train.py --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## References

- [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/2000/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the original Gym environments
- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework
- The reinforcement learning community for research and algorithms

 
# Actor-Critic-Reinforcement-Learning-Implementation
