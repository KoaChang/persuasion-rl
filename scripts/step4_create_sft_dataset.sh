#!/bin/bash
# Step 4: Create SFT dataset
# Creates the final training dataset from preprocessed examples
# Default: 11.75k total (9.4k SFT, 2.15k RLAIF, 200 RLHF, 115 final eval)

set -e

# Activate virtual environment if not already activated
# if [[ -z "$VIRTUAL_ENV" ]]; then
#     echo "Activating virtual environment..."
#     source venv/bin/activate
# fi

echo "================================================"
echo "Step 4: Creating SFT dataset..."
echo "================================================"
echo "Creating 11.75k total: 9.4k SFT, 2.15k RLAIF, 200 RLHF, 115 final eval"
echo ""

# Default: Use CMV only
# To use P4G, add --use-p4g and --p4g-file flags
# Reserves 115 examples for final evaluation by default (--reserve-final-eval 115)
python src/data/create_sft_dataset.py \
    --cmv-file data/processed/cmv_examples.jsonl \
    --use-cmv \
    --reserve-final-eval 115 \
    --output-dir data/processed

echo ""
echo "================================================"
echo "Step 4 complete!"
echo "================================================"
echo "Dataset files saved to data/processed/:"
echo "  - sft_train.jsonl (training data)"
echo "  - sft_test.jsonl (evaluation data)"
echo "  - rlaif_pool.jsonl (for RLAIF)"
echo "  - rlhf_pool.jsonl (for RLHF)"
echo ""
echo "Next step: Run step5_train_sft.sh"
echo ""

