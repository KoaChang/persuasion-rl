#!/bin/bash
# Step 2: Preprocess CMV dataset
# Preprocesses the raw CMV data into a format suitable for training

set -e

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "================================================"
echo "Step 2: Preprocessing CMV dataset..."
echo "================================================"

python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl

echo ""
echo "================================================"
echo "Step 2 complete!"
echo "================================================"
echo "Preprocessed CMV data saved to: data/processed/cmv_examples.jsonl"
echo ""
echo "Next step: Run step4_create_sft_dataset.sh"
echo "  (or step3_preprocess_p4g.sh if you want to use PersuasionForGood)"
echo ""

