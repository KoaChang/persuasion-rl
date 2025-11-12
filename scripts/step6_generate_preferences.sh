#!/bin/bash
# Step 6: Generate preference data
# Generates preference pairs for RLHF using the trained SFT model

set -e

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

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
echo "Step 6 complete!"
echo "================================================"
echo "Preference data saved to: data/preferences/"
echo ""
echo "================================================"
echo "All steps complete!"
echo "================================================"
echo ""

