#!/bin/bash
# Run complete evaluation pipeline
#
# This script:
#   1. Evaluates all 4 models on test set
#   2. Analyzes results with statistical tests
#   3. Generates visualizations and qualitative examples
#
# Usage:
#   bash scripts/run_evaluation.sh

set -e  # Exit on error

echo "=========================================="
echo "Running Evaluation Pipeline"
echo "=========================================="
echo ""

# Check if trained models exist
if [ ! -d "models/checkpoints/qwen-sft/final" ]; then
    echo "Error: SFT model not found!"
    echo "Path: models/checkpoints/qwen-sft/final"
    exit 1
fi

# Check for API keys
if ! grep -q "sk-" configs/api_config.yaml 2>/dev/null; then
    echo "Warning: API keys may not be set in configs/api_config.yaml"
    echo "Please ensure you have filled in your Claude and OpenAI API keys."
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Evaluate all models
echo "=========================================="
echo "[1/2] Evaluating all models"
echo "=========================================="
echo "This will:"
echo "  - Load all available models"
echo "  - Generate responses for 115 test examples"
echo "  - Rank with Claude 4.5 Sonnet grader"
echo "  - Compute similarity to oracle"
echo ""
echo "Estimated time: 30-60 minutes"
echo "Estimated cost: ~$5-10 in API calls"
echo ""
read -p "Press Enter to continue..."

python src/eval/evaluate_all_models.py

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 2: Analyze results
echo "=========================================="
echo "[2/2] Analyzing results"
echo "=========================================="
echo "This will:"
echo "  - Perform statistical analysis"
echo "  - Generate visualizations"
echo "  - Extract qualitative examples"
echo ""

python src/eval/analyze_results.py

if [ $? -ne 0 ]; then
    echo "Error: Analysis failed!"
    exit 1
fi

echo ""
echo "✓ Analysis complete!"
echo ""

# Summary
echo "=========================================="
echo "✓ Evaluation pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/final_evaluation.json (full results)"
echo "  - results/analysis_report.txt (summary)"
echo "  - results/qualitative_examples.txt (examples)"
echo "  - results/figures/ (visualizations)"
echo ""
echo "To view results:"
echo "  cat results/analysis_report.txt"
echo "  open results/figures/grader_scores.png"
echo ""
