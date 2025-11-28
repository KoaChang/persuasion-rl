#!/bin/bash
# Check setup and prerequisites for RLAIF/RLHF pipeline

echo "=========================================="
echo "RLAIF/RLHF Pipeline Setup Check"
echo "=========================================="
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"
if [[ "$python_version" < "3.10" ]]; then
    echo "  ⚠️  Warning: Python 3.10+ recommended"
else
    echo "  ✓ Python version OK"
fi
echo ""

# Check required packages
echo "[2/7] Checking required packages..."
required_packages=("torch" "transformers" "datasets" "peft" "trl" "anthropic" "openai" "scipy" "sklearn" "matplotlib")
missing_packages=()

for package in "${required_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "  ✓ $package"
    else
        echo "  ✗ $package (missing)"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -eq 0 ]; then
    echo "  ✓ All required packages installed"
else
    echo ""
    echo "  ⚠️  Missing packages: ${missing_packages[*]}"
    echo "  Install with: pip install -r requirements_rlaif_rlhf.txt"
fi
echo ""

# Check API keys
echo "[3/7] Checking API keys..."
if [ -f "configs/api_config.yaml" ]; then
    echo "  ✓ API config file exists"

    # Check for placeholder keys
    if grep -q "YOUR_CLAUDE_API_KEY\|YOUR_OPENAI_API_KEY\|api_key: \"\"" configs/api_config.yaml; then
        echo "  ⚠️  API keys not set (still have placeholders)"
        echo "     Please edit configs/api_config.yaml"
    elif grep -q "sk-\|claude-" configs/api_config.yaml; then
        echo "  ✓ API keys appear to be set"
    else
        echo "  ⚠️  Cannot verify API keys"
    fi
else
    echo "  ✗ API config file not found"
fi
echo ""

# Check SFT model
echo "[4/7] Checking SFT model..."
if [ -d "models/checkpoints/qwen-sft/final" ]; then
    echo "  ✓ SFT model exists"
    model_size=$(du -sh models/checkpoints/qwen-sft/final | awk '{print $1}')
    echo "    Size: $model_size"
else
    echo "  ✗ SFT model not found"
    echo "    Expected: models/checkpoints/qwen-sft/final"
fi
echo ""

# Check preference data
echo "[5/7] Checking preference data..."
if [ -f "data/preferences/ai_pool_responses.jsonl" ]; then
    ai_count=$(wc -l < data/preferences/ai_pool_responses.jsonl)
    echo "  ✓ AI pool responses: $ai_count pairs"
else
    echo "  ✗ AI pool responses not found"
fi

if [ -f "data/preferences/human_pool_responses.jsonl" ]; then
    human_count=$(wc -l < data/preferences/human_pool_responses.jsonl)
    echo "  ✓ Human pool responses: $human_count pairs"
else
    echo "  ✗ Human pool responses not found"
fi
echo ""

# Check labeled preferences
echo "[6/7] Checking labeled preferences..."
if [ -f "data/preferences/ai_pool_labeled.jsonl" ]; then
    labeled_count=$(wc -l < data/preferences/ai_pool_labeled.jsonl)
    echo "  ✓ AI preferences labeled: $labeled_count / 2150"
else
    echo "  ⚠️  AI preferences not yet labeled"
    echo "     Run: python src/preferences/label_ai_preferences.py"
fi

if [ -f "data/preferences/human_pool_labeled.jsonl" ]; then
    labeled_count=$(wc -l < data/preferences/human_pool_labeled.jsonl)
    echo "  ✓ Human preferences labeled: $labeled_count / 200"
else
    echo "  ⚠️  Human preferences not yet labeled"
    echo "     Run: python src/preferences/label_human_preferences.py"
fi
echo ""

# Check trained models
echo "[7/7] Checking trained DPO models..."
if [ -d "models/checkpoints/qwen-rlaif/final" ]; then
    echo "  ✓ RLAIF model trained"
else
    echo "  ⚠️  RLAIF model not yet trained"
fi

if [ -d "models/checkpoints/qwen-rlhf/final" ]; then
    echo "  ✓ RLHF model trained"
else
    echo "  ⚠️  RLHF model not yet trained"
fi

if [ -d "models/checkpoints/qwen-rlaif-rlhf/final" ]; then
    echo "  ✓ RLAIF+RLHF model trained"
else
    echo "  ⚠️  RLAIF+RLHF model not yet trained"
fi
echo ""

# GPU check
echo "=========================================="
echo "GPU Check"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  ⚠️  No NVIDIA GPU detected"
    echo "     Training will be slow without GPU"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="

if [ ${#missing_packages[@]} -eq 0 ] && [ -d "models/checkpoints/qwen-sft/final" ]; then
    echo "✓ Environment is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Set API keys in configs/api_config.yaml"
    echo "  2. Run: python src/preferences/label_ai_preferences.py"
    echo "  3. See QUICK_START.md for full pipeline"
else
    echo "⚠️  Setup incomplete"
    echo ""
    echo "Please complete:"
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo "  - Install missing packages: pip install -r requirements_rlaif_rlhf.txt"
    fi
    if [ ! -d "models/checkpoints/qwen-sft/final" ]; then
        echo "  - Train SFT model first"
    fi
fi
echo ""
