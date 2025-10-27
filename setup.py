#!/usr/bin/env python3
"""
Setup script for the RL implementation project.

This script helps set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install project dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Check if PyTorch is installed correctly
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed successfully")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available, using CPU")
    except ImportError:
        print("❌ PyTorch installation failed")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["models", "logs", "plots", "checkpoints"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run basic tests to verify installation."""
    if not run_command("python test_implementation.py", "Running basic tests"):
        return False
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up RL Implementation Project\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Tests failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python train.py --env CartPole-v1 --episodes 100")
    print("2. Try advanced training: python train_advanced.py --algorithm ppo --env CartPole-v1")
    print("3. Launch web interface: streamlit run app.py")
    print("4. Open Jupyter notebook: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    main()
