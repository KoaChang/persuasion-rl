#!/bin/bash
# Setup environment for SFT training

set -e

echo "================================================"
echo "Setting up environment for Persuasion-RL SFT"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Environment setup complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download datasets: python src/data/download_datasets.py"
echo "  2. Preprocess CMV: python src/data/preprocess_cmv.py"
echo "  3. Preprocess P4G: python src/data/preprocess_persuasionforgood.py"
echo "  4. Create SFT dataset: python src/data/create_sft_dataset.py"
echo "  5. Train SFT model: python src/sft/train_sft.py --config configs/sft_config.yaml"
echo ""

