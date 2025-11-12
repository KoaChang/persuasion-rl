#!/bin/bash
# Step 1: Download datasets
# Downloads the CMV dataset (P4G disabled by default)

set -e

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "================================================"
echo "Step 1: Downloading datasets..."
echo "================================================"
echo "Note: Downloading CMV only (P4G download disabled by default)"

python src/data/download_datasets.py --output-dir data/raw --dataset cmv

echo ""
echo "================================================"
echo "Step 1 complete!"
echo "================================================"
echo "CMV dataset downloaded to: data/raw/cmv"
echo ""
echo "Next step: Run step2_preprocess_cmv.sh"
echo ""

