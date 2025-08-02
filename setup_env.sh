#!/bin/bash
# Environment setup script for Lie-Circuit experiment
# Step 0: env-setup

set -e  # Exit on error

echo "=== Lie-Circuit Environment Setup ==="
echo "Starting at $(date)"

# Create conda environment
echo "Creating conda environment 'lie_circuit' with Python 3.11..."
conda create -n lie_circuit python=3.11 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lie_circuit

# Install required packages
echo "Installing required packages..."
pip install transformerlens==1.11.0 torch==2.2.1 numpy pandas tqdm matplotlib
pip install hydra-core openai

# Note: open-cross-layer-transcoder==0.1.3 may not exist on PyPI
# For now, we'll create a placeholder or use an alternative
# pip install open-cross-layer-transcoder==0.1.3

# Create budget log file
touch budget.log

# Run GPU smoke test
echo "Running GPU smoke test..."
python gpu_smoke_test.py

# Check Lambda credits (placeholder - adjust based on actual Lambda CLI)
# lamctl credits >> budget.log
echo "$(date): Credits check placeholder" >> budget.log

echo "Environment setup completed at $(date)"
exit 0