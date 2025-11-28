# RLAIF and RLHF Training Pipeline

This README provides step-by-step instructions for running the RLAIF and RLHF portions of the CS230 persuasion-RL project.

## Overview

This pipeline trains and evaluates 4 models:

1. **Base**: Qwen2.5-0.5B-Instruct (no fine-tuning)
2. **SFT**: Base + supervised fine-tuning on persuasion dialogues (already trained)
3. **RLHF-only**: SFT + DPO on 180 human-labeled preferences
4. **RLAIF+RLHF**: SFT + DPO on 2,150 AI-labeled preferences + DPO on 180 human-labeled preferences

## Prerequisites

✅ **Already Completed:**

- Data loading and preprocessing
- SFT model training (checkpoint in `models/checkpoints/qwen-sft/final`)
- Preference pair generation (2,150 AI pool + 200 human pool)

⚠️ **Required:**

- Claude API key (for AI grading and evaluation)
- OpenAI API key (for similarity embeddings)
- GPU access (g5.xlarge or similar)
- ~$10-30 budget for API calls

## Quick Start

### Step 0: Set Up API Keys

1. Open `configs/api_config.yaml`
2. Fill in your API keys:

```yaml
anthropic:
  api_key: "YOUR_CLAUDE_API_KEY_HERE"

openai:
  api_key: "YOUR_OPENAI_API_KEY_HERE"
```

### Step 1: Label AI Preferences (~2-3 hours)

```bash
python src/preferences/label_ai_preferences.py
```

This will:

- Label 2,150 preference pairs using Claude 4.5 Sonnet
- Cost: ~$3-10 in API calls
- Save results to `data/preferences/ai_pool_labeled.jsonl`
- Support resume with `--resume` flag if interrupted

**Testing with small sample:**

```bash
python src/preferences/label_ai_preferences.py --max_examples 10
```

### Step 2: Label Human Preferences (~3-6 hours of your time)

```bash
python src/preferences/label_human_preferences.py
```

This will:

- Show you 200 preference pairs in an interactive CLI
- You choose which response is more persuasive (press 1 or 2)
- Save results to `data/preferences/human_pool_labeled.jsonl`
- Support resume with `--resume` flag

**Tips for human labeling:**

- Take breaks every 50 examples to avoid fatigue
- Use `Q` to quit and save progress anytime
- Use `--resume` to continue from where you left off

### Step 3: Validate Preferences

```bash
python src/preferences/validate_preferences.py
```

This will:

- Validate labeled data for errors and biases
- Create train/val splits (90/10)
- Generate validation report
- Save splits to `data/preferences/rlaif_train.jsonl`, `rlhf_train.jsonl`, etc.

### Step 4: Train DPO Models (~3.5-5 hours on g5.xlarge)

**Option A: Train all models in sequence**

```bash
bash scripts/train_all_dpo.sh
```

**Option B: Train individually**

```bash
# RLAIF (SFT → DPO on AI preferences) - ~2-3 hours
python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlaif

# RLHF-only (SFT → DPO on human preferences) - ~30-45 min
python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlhf

# RLAIF+RLHF (RLAIF → DPO on human preferences) - ~30-45 min
python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlaif_to_rlhf
```

**Models will be saved to:**

- `models/checkpoints/qwen-rlaif/final`
- `models/checkpoints/qwen-rlhf/final`
- `models/checkpoints/qwen-rlaif-rlhf/final`

### Step 5: Evaluate All Models (~30-60 minutes)

```bash
python src/eval/evaluate_all_models.py
```

This will:

- Load all 4 models (base, SFT, RLHF, RLAIF+RLHF)
- Generate responses for 115 test examples
- Rank responses using Claude 4.5 Sonnet grader
- Compute similarity to oracle using OpenAI embeddings
- Save results to `results/final_evaluation.json`

**Testing with small sample:**

```bash
python src/eval/evaluate_all_models.py --max_examples 5
```

### Step 6: Analyze Results

```bash
python src/eval/analyze_results.py
```

This will:

- Perform statistical analysis (bootstrap CI, t-tests)
- Check for position bias
- Compute pairwise win rates
- Generate visualizations (bar charts, distributions, heatmaps)
- Extract qualitative examples

**Outputs:**

- `results/analysis_report.txt` - Statistical summary
- `results/qualitative_examples.txt` - Best/worst examples per model
- `results/figures/grader_scores.png` - Bar chart of grader scores
- `results/figures/similarity_distributions.png` - Similarity histograms
- `results/figures/pairwise_win_rates.png` - Win rate heatmap

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PREFERENCE DATA PREPARATION                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Label AI preferences (Claude API)         ~2-3 hours     │
│ 2. Label human preferences (manual)          ~3-6 hours     │
│ 3. Validate and create splits                ~5 minutes     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: DPO TRAINING                                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Train RLAIF model                          ~2-3 hours     │
│ 2. Train RLHF-only model                      ~30-45 min     │
│ 3. Train RLAIF+RLHF model                     ~30-45 min     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: EVALUATION                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Generate responses from all models        ~30-60 min     │
│ 2. Rank with Claude grader                                  │
│ 3. Compute similarity to oracle                             │
│ 4. Analyze results and visualize             ~5 minutes     │
└─────────────────────────────────────────────────────────────┘
```

**Total Time:** ~9-14 hours (including labeling time)

## Project Structure

```
persuasion-rl/
├── configs/
│   ├── api_config.yaml              # API keys (you fill in)
│   ├── dpo_config.yaml              # DPO hyperparameters
│   └── eval_config.yaml             # Evaluation settings
│
├── data/
│   └── preferences/
│       ├── ai_pool_responses.jsonl  # [Existing] Unlabeled AI pairs
│       ├── human_pool_responses.jsonl # [Existing] Unlabeled human pairs
│       ├── ai_pool_labeled.jsonl    # [Generated] Labeled AI pairs
│       ├── human_pool_labeled.jsonl # [Generated] Labeled human pairs
│       ├── rlaif_train.jsonl        # [Generated] RLAIF training data
│       ├── rlaif_val.jsonl          # [Generated] RLAIF validation data
│       ├── rlhf_train.jsonl         # [Generated] RLHF training data
│       └── rlhf_val.jsonl           # [Generated] RLHF validation data
│
├── models/
│   └── checkpoints/
│       ├── qwen-sft/final/          # [Existing] SFT model
│       ├── qwen-rlaif/final/        # [Generated] RLAIF model
│       ├── qwen-rlhf/final/         # [Generated] RLHF model
│       └── qwen-rlaif-rlhf/final/   # [Generated] RLAIF+RLHF model
│
├── src/
│   ├── preferences/
│   │   ├── label_ai_preferences.py  # AI preference labeling
│   │   ├── label_human_preferences.py # Human preference labeling
│   │   └── validate_preferences.py  # Validation and splitting
│   ├── dpo/
│   │   └── train_dpo.py             # DPO training script
│   ├── eval/
│   │   ├── grader.py                # Grader utilities
│   │   ├── evaluate_all_models.py   # Main evaluation script
│   │   └── analyze_results.py       # Statistical analysis
│   └── utils/
│       └── api_clients.py           # Claude and OpenAI clients
│
├── scripts/
│   └── train_all_dpo.sh             # Run all DPO training
│
└── results/                          # [Generated] Evaluation outputs
    ├── final_evaluation.json
    ├── analysis_report.txt
    ├── qualitative_examples.txt
    └── figures/
```

## Configuration Files

### `configs/dpo_config.yaml`

Key hyperparameters you can tune:

```yaml
dpo:
  beta: 0.1 # KL penalty (0.1-0.5, higher = stay closer to reference)

rlaif:
  learning_rate: 5.0e-5
  num_epochs: 1 # Large dataset, 1 epoch may be enough

rlhf:
  learning_rate: 2.0e-5 # Lower LR for small dataset
  num_epochs: 3 # More epochs for small dataset

rlaif_to_rlhf:
  learning_rate: 2.0e-5
  num_epochs: 3
```

### `configs/eval_config.yaml`

Evaluation settings:

```yaml
generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9

oracle:
  generate: true # Generate oracle responses with Claude

evaluation:
  test_file: "./data/processed/final_eval_reserved.jsonl"
  randomize_positions: true # Avoid position bias
```

## Troubleshooting

### API Errors

**"API key not set"**

- Fill in your API keys in `configs/api_config.yaml`

**Rate limit errors**

- The scripts have built-in retry logic
- Adjust `rate_limiting.requests_per_minute` in `configs/api_config.yaml`

**API timeouts**

- Increase `openai.timeout` in `configs/api_config.yaml`
- Check your internet connection

### Training Errors

**Out of memory (OOM)**

- Reduce `per_device_train_batch_size` in `configs/dpo_config.yaml`
- Enable 4-bit quantization (edit `train_dpo.py` to use QLoRA)

**Model checkpoint not found**

- Ensure SFT model exists in `models/checkpoints/qwen-sft/final`
- For RLAIF+RLHF, ensure RLAIF model was trained first

**NaN loss or training divergence**

- Reduce learning rate
- Increase `beta` (KL penalty) in DPO config
- Check for corrupted preference data

### Evaluation Errors

**Model not found**

- Ensure all DPO models are trained before evaluation
- The script will skip missing models with a warning

**Grader parsing errors**

- Claude may occasionally output unexpected formats
- The script will log these to `.errors` file and continue

## Cost Estimates

### API Costs

**AI Preference Labeling:**

- 2,150 pairs × ~500 tokens/pair = ~1M tokens
- Claude Sonnet 4.5: ~$3-5

**Evaluation:**

- 115 examples × 4 models = 460 generation requests
- 115 rankings = 115 × ~800 tokens = ~92K tokens
- 115 × 4 similarity computations = 460 embeddings
- Total: ~$5-10

**Total API costs: ~$10-30**

### Compute Costs

**DPO Training on g5.xlarge ($1.006/hour):**

- RLAIF: 2-3 hours = $2-3
- RLHF: 0.5-0.75 hours = $0.50-0.75
- RLAIF+RLHF: 0.5-0.75 hours = $0.50-0.75
- Total: ~$3-5

**Evaluation: ~$0.50-1**

**Total compute costs: ~$4-6**

## Expected Results

### Success Criteria

✅ **Minimum viable results:**

- All 4 models trained and saved
- Evaluation completes on 115 test examples
- Grader scores show improvement: RLHF/RLAIF > SFT > Base
- No catastrophic failures (gibberish generation)

✅ **Strong results:**

- RLAIF+RLHF > RLHF-only (showing warmup benefits)
- Similarity scores correlate with grader rankings
- Statistical significance (p < 0.05) in comparisons
- Qualitative examples show clear improvements

### Example Output

```
EVALUATION SUMMARY
========================================
Grader Ranking Scores (normalized to 0-1):
  rlaif_rlhf     : 0.687
  rlhf           : 0.652
  sft            : 0.478
  base           : 0.183

Average Similarity to Oracle:
  rlaif_rlhf     : 0.842
  rlhf           : 0.831
  sft            : 0.791
  base           : 0.723
```

## Tips for CS230 Report

Key points to highlight:

1. **Novel contribution**: Combining RLAIF warmup with limited RLHF
2. **Efficiency**: Large AI-labeled dataset + small human-labeled dataset
3. **Methodology**: DPO instead of PPO (simpler, more stable)
4. **Evaluation**: Dual metrics (grader + similarity)
5. **Results**: Show quantitative improvements and qualitative examples

6. **RLAIF warmup strategy**: Large AI-graded dataset helps when human labels are scarce
7. **DPO vs PPO**: Why DPO is simpler and more stable for small datasets
8. **Reference model choice**: Using SFT as reference for all stages (Option A)
9. **Evaluation metrics**: Both grader rankings and similarity to oracle

### Figures to Include

1. Bar chart of grader scores with confidence intervals
2. Similarity score distributions
3. Pairwise win rate heatmap
4. Qualitative examples showing improvements

### Discussion Topics

- Did RLAIF+RLHF outperform RLHF-only? If so, by how much?
- Position bias check: was the grader consistent?
- Safety considerations: did models become more manipulative?
- Limitations: small eval set (115 examples), single domain (persuasion)

## Next Steps

After completing this pipeline, consider:

1. **Hyperparameter tuning**: Adjust beta, learning rates, epochs
2. **Error analysis**: Manually inspect failure cases
3. **Alternative metrics**: Human evaluation, ROUGE scores
4. **Ablation studies**: Try different reference models, larger datasets

## Support

For issues or questions:

- Check the detailed implementation plan in `.claude/tasks/rlaif_rlhf_implementation_plan.md`
- Review the validation report in `data/preferences/validation_report.txt`
- Check error logs in `*.errors` files

Good luck with your CS230 project! 🚀
