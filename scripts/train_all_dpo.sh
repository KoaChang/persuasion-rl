#!/bin/bash
# Train all DPO models in sequence
#
# This script trains three DPO models:
#   1. RLAIF: SFT → DPO on AI preferences (~2-3 hours)
#   2. RLHF-only: SFT → DPO on human preferences (~30-45 min)
#   3. RLAIF→RLHF: RLAIF → DPO on human preferences (~30-45 min)
#
# Total estimated time: ~3.5-5 hours on g5.xlarge
#
# Usage:
#   bash scripts/train_all_dpo.sh

set -e  # Exit on error

echo "=========================================="
echo "Starting DPO Training Pipeline"
echo "=========================================="
echo ""

# Check if labeled preference data exists
if [ ! -f "data/preferences/rlaif_train.jsonl" ]; then
    echo "Error: RLAIF training data not found!"
    echo "Please run preference labeling and validation first:"
    echo "  1. python src/preferences/label_ai_preferences.py"
    echo "  2. python src/preferences/label_human_preferences.py"
    echo "  3. python src/preferences/validate_preferences.py"
    exit 1
fi

if [ ! -f "data/preferences/rlhf_train.jsonl" ]; then
    echo "Error: RLHF training data not found!"
    echo "Please run preference labeling and validation first."
    exit 1
fi

# Stage 1: RLAIF (SFT → DPO on AI preferences)
echo "=========================================="
echo "[1/3] Training RLAIF model"
echo "=========================================="
echo "Starting from: SFT model"
echo "Training on: ~1,935 AI-labeled preference pairs"
echo "Estimated time: 2-3 hours"
echo ""
python src/dpo/train_dpo.py \
    --config configs/dpo_config.yaml \
    --stage rlaif

echo ""
echo "✓ RLAIF training complete!"
echo ""

# Stage 2: RLHF-only (SFT → DPO on human preferences)
echo "=========================================="
echo "[2/3] Training RLHF-only model"
echo "=========================================="
echo "Starting from: SFT model"
echo "Training on: ~180 human-labeled preference pairs"
echo "Estimated time: 30-45 minutes"
echo ""
python src/dpo/train_dpo.py \
    --config configs/dpo_config.yaml \
    --stage rlhf

echo ""
echo "✓ RLHF-only training complete!"
echo ""

# Stage 3: RLAIF→RLHF (RLAIF → DPO on human preferences)
echo "=========================================="
echo "[3/3] Training RLAIF→RLHF model"
echo "=========================================="
echo "Starting from: RLAIF model"
echo "Training on: ~180 human-labeled preference pairs"
echo "Estimated time: 30-45 minutes"
echo ""
python src/dpo/train_dpo.py \
    --config configs/dpo_config.yaml \
    --stage rlaif_to_rlhf

echo ""
echo "✓ RLAIF→RLHF training complete!"
echo ""

# Final summary
echo "=========================================="
echo "✓ All DPO training complete!"
echo "=========================================="
echo ""
echo "Models saved to:"
echo "  1. models/checkpoints/qwen-rlaif/final"
echo "  2. models/checkpoints/qwen-rlhf/final"
echo "  3. models/checkpoints/qwen-rlaif-rlhf/final"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: python src/eval/evaluate_all_models.py"
echo "  2. Analyze results: python src/eval/analyze_results.py"
echo ""
