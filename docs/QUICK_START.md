# Quick Start Guide - RLAIF & RLHF Pipeline

## Installation

```bash
# Install additional dependencies
pip install -r requirements_rlaif_rlhf.txt
```

## Setup (5 minutes)

1. **Add API Keys** to `configs/api_config.yaml`:
   ```yaml
   anthropic:
     api_key: "YOUR_CLAUDE_API_KEY"
   openai:
     api_key: "YOUR_OPENAI_API_KEY"
   ```

## Run Full Pipeline

### Option 1: Complete Automated Pipeline

```bash
# 1. Label AI preferences (~2-3 hours, $3-10)
python src/preferences/label_ai_preferences.py

# 2. Label human preferences (~3-6 hours of your time)
python src/preferences/label_human_preferences.py

# 3. Validate preferences (~5 minutes)
python src/preferences/validate_preferences.py

# 4. Train all DPO models (~3.5-5 hours)
bash scripts/train_all_dpo.sh

# 5. Evaluate and analyze (~30-60 minutes)
bash scripts/run_evaluation.sh
```

### Option 2: Test with Small Samples First

```bash
# Test AI labeling with 10 examples
python src/preferences/label_ai_preferences.py --max_examples 10

# Test human labeling with 10 examples
python src/preferences/label_human_preferences.py
# (label 10 examples, then press Q to quit)

# Validate
python src/preferences/validate_preferences.py

# Test evaluation with 5 examples
python src/eval/evaluate_all_models.py --max_examples 5

# Analyze results
python src/eval/analyze_results.py
```

## Expected Timeline

| Step | Time | Cost |
|------|------|------|
| AI preference labeling | 2-3 hours | $3-10 |
| Human preference labeling | 3-6 hours | $0 |
| Validation | 5 minutes | $0 |
| DPO training | 3.5-5 hours | $3-5 |
| Evaluation | 30-60 minutes | $5-10 |
| **Total** | **~9-14 hours** | **~$11-25** |

## Checking Progress

```bash
# Check if AI preferences are labeled
ls -lh data/preferences/ai_pool_labeled.jsonl

# Check if human preferences are labeled
ls -lh data/preferences/human_pool_labeled.jsonl

# Check if splits are created
ls -lh data/preferences/rlaif_train.jsonl
ls -lh data/preferences/rlhf_train.jsonl

# Check if models are trained
ls -lh models/checkpoints/qwen-rlaif/final/
ls -lh models/checkpoints/qwen-rlhf/final/
ls -lh models/checkpoints/qwen-rlaif-rlhf/final/

# Check if evaluation is complete
ls -lh results/final_evaluation.json
```

## Troubleshooting

### Common Issues

**"API key not set"**
```bash
# Edit config file
nano configs/api_config.yaml
```

**Out of memory during training**
```bash
# Reduce batch size in configs/dpo_config.yaml
# Change per_device_train_batch_size from 4 to 2
```

**Resume interrupted labeling**
```bash
# AI labeling
python src/preferences/label_ai_preferences.py --resume

# Human labeling
python src/preferences/label_human_preferences.py --resume
```

**Evaluation fails to find models**
```bash
# Make sure training completed successfully
ls models/checkpoints/*/final/

# Evaluation will skip missing models with a warning
```

## Viewing Results

```bash
# View summary
cat results/analysis_report.txt

# View qualitative examples
less results/qualitative_examples.txt

# View plots (macOS)
open results/figures/grader_scores.png
open results/figures/similarity_distributions.png
open results/figures/pairwise_win_rates.png

# View plots (Linux with display)
xdg-open results/figures/grader_scores.png
```

## Next Steps

After completing the pipeline:

1. **Review results** in `results/analysis_report.txt`
2. **Check visualizations** in `results/figures/`
3. **Read qualitative examples** in `results/qualitative_examples.txt`
4. **Write up findings** for CS230 report

For detailed information, see `RLAIF_RLHF_README.md`.
