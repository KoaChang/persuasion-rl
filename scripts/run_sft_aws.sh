#!/bin/bash
# Run SFT training on AWS instance
# This script should be run on an AWS g5.xlarge or similar GPU instance

set -e

echo "================================================"
echo "SFT Training on AWS"
echo "================================================"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo ""
else
    echo "Warning: nvidia-smi not found. Make sure you're on a GPU instance."
fi

# Setup environment
echo "Setting up environment..."
bash scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate

# Login to wandb (you'll need to provide your API key)
echo ""
echo "================================================"
echo "Weights & Biases Setup"
echo "================================================"
echo "Please login to Weights & Biases:"
wandb login

# Download and preprocess data
echo ""
echo "================================================"
echo "Step 1: Downloading datasets..."
echo "================================================"
echo "Note: Downloading CMV only (P4G download disabled by default)"
python src/data/download_datasets.py --output-dir data/raw --dataset cmv

echo ""
echo "================================================"
echo "Step 2: Preprocessing CMV dataset..."
echo "================================================"
python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl

echo ""
echo "================================================"
echo "Step 3: Skipping PersuasionForGood (not used by default)..."
echo "================================================"
echo "To enable P4G, use --use-p4g flag in create_sft_dataset.py"
# Uncomment below to preprocess P4G if needed:
# python src/data/preprocess_persuasionforgood.py \
#     --input-dir data/raw/persuasionforgood \
#     --output-file data/processed/p4g_examples.jsonl

echo ""
echo "================================================"
echo "Step 4: Creating SFT dataset (50k examples from CMV)..."
echo "================================================"
python src/data/create_sft_dataset.py \
    --cmv-file data/processed/cmv_examples.jsonl \
    --use-cmv \
    --max-examples 50000 \
    --output-dir data/processed

echo ""
echo "================================================"
echo "Step 5: Training SFT model (on 50k CMV examples)..."
echo "================================================"
python src/sft/train_sft.py --config configs/sft_config.yaml

echo ""
echo "================================================"
echo "Step 6: Generating preference data..."
echo "================================================"
python src/sft/generate_preferences.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/sft_test.jsonl \
    --output-dir data/preferences

echo ""
echo "================================================"
echo "All steps complete!"
echo "================================================"
echo ""
echo "Model saved to: models/checkpoints/qwen-sft/final"
echo "Preference data saved to: data/preferences/"
echo ""

