#!/bin/bash
# Step 3: Preprocess PersuasionForGood dataset (OPTIONAL)
# This step is optional and only needed if you want to use P4G data
# Note: You must download P4G first with: python src/data/download_datasets.py --dataset p4g

set -e

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "================================================"
echo "Step 3: Preprocessing PersuasionForGood dataset..."
echo "================================================"
echo "Note: This is an OPTIONAL step"
echo ""

# Check if P4G data exists
if [ ! -d "data/raw/persuasionforgood" ]; then
    echo "Error: P4G data not found at data/raw/persuasionforgood"
    echo "Please download it first with:"
    echo "  python src/data/download_datasets.py --output-dir data/raw --dataset p4g"
    exit 1
fi

python src/data/preprocess_persuasionforgood.py \
    --input-dir data/raw/persuasionforgood \
    --output-file data/processed/p4g_examples.jsonl

echo ""
echo "================================================"
echo "Step 3 complete!"
echo "================================================"
echo "Preprocessed P4G data saved to: data/processed/p4g_examples.jsonl"
echo ""
echo "Next step: Run step4_create_sft_dataset.sh with --use-p4g flag"
echo ""

