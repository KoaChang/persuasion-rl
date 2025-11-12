#!/bin/bash
# Step 5: Train SFT model
# Trains the supervised fine-tuning model using the prepared dataset
# Training uses ~9.4k CMV examples (80% of 11.75k total)

set -e

# Activate virtual environment if not already activated
# if [[ -z "$VIRTUAL_ENV" ]]; then
#     echo "Activating virtual environment..."
#     source venv/bin/activate
# fi

echo "================================================"
echo "Step 5: Training SFT model..."
echo "================================================"
echo "Training on 9.4k CMV examples (80% of 11.75k total)"
echo "This will take a while..."
echo ""

python src/sft/train_sft.py --config configs/sft_config.yaml

echo ""
echo "================================================"
echo "Step 5 complete!"
echo "================================================"
echo "Model saved to: models/checkpoints/qwen-sft/final"
echo ""
echo "Next step: Run step6_generate_preferences.sh"
echo ""

