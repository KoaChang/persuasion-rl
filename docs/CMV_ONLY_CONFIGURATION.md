# CMV-Only Configuration

## Overview

By default, this project uses **only the CMV (ChangeMyView) dataset** for SFT training, with **30,000 examples**.

The PersuasionForGood dataset has been disabled by default based on quality concerns, but the preprocessing code remains available if you want to re-enable it.

## Current Configuration

### Default Settings

- **Dataset**: 100% CMV (ChangeMyView)
- **Size**: 30,000 examples
- **Source**: Delta-winning comments from Reddit's r/ChangeMyView
- **Quality**: High-quality persuasive arguments with verified effectiveness

### Why CMV Only?

1. **Higher Quality**: CMV comments won deltas (changed someone's view) - verified persuasion success
2. **Better Context**: Longer, more nuanced arguments
3. **Natural Language**: Reddit-style conversational tone
4. **Diverse Topics**: Wide range of subjects and argument styles
5. **Consistent Format**: All examples follow similar structure

### Why 30k Examples?

Given your RL plans (300 RLHF, 10k RLAIF):

- **100x your RLHF data** (300 prompts)
- **3x your RLAIF data** (10k prompts)
- **Minimum viable size** for strong SFT performance
- **Cost efficient**: ~6-8 hours training, ~$6-8
- **Still allows RL improvement**: Not so good that DPO can't help

## Files Modified

### 1. `configs/data_config.yaml`

```yaml
datasets:
  use_cmv: true # Use CMV dataset
  use_p4g: false # Skip PersuasionForGood by default
  max_examples: 30000 # Default to 30k examples
```

### 2. `src/data/create_sft_dataset.py`

New flags:

- `--use-cmv` / `--no-cmv`: Control CMV inclusion (default: True)
- `--use-p4g`: Enable PersuasionForGood (default: False)
- `--max-examples 30000`: Default limit (was: no limit)

### 3. `scripts/run_sft_aws.sh`

- Downloads CMV only
- Skips P4G preprocessing
- Uses 30k examples by default

## How to Use

### Default Usage (CMV Only, 30k)

Just run the standard pipeline:

```bash
bash scripts/run_sft_aws.sh
```

This will:

1. Download CMV dataset only
2. Preprocess CMV examples
3. Create SFT dataset with 30k examples from CMV
4. Train model
5. Generate preference data

### Manual Step-by-Step (CMV Only)

```bash
# 1. Download CMV
python src/data/download_datasets.py --output-dir data/raw --dataset cmv

# 2. Preprocess CMV
python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl

# 3. Create SFT dataset (30k from CMV)
python src/data/create_sft_dataset.py \
    --cmv-file data/processed/cmv_examples.jsonl \
    --use-cmv \
    --max-examples 30000 \
    --output-dir data/processed

# 4. Train
python src/sft/train_sft.py --config configs/sft_config.yaml
```

## Customization Options

### Change Dataset Size

Use different amounts of CMV data:

```bash
# Use 20k examples (faster, cheaper)
python src/data/create_sft_dataset.py \
    --use-cmv \
    --max-examples 20000

# Use 50k examples (stronger base, more expensive)
python src/data/create_sft_dataset.py \
    --use-cmv \
    --max-examples 50000

# Use all available CMV data (no limit)
python src/data/create_sft_dataset.py \
    --use-cmv \
    --max-examples 0
```

### Re-enable PersuasionForGood

If you want to try P4G despite quality concerns:

```bash
# Download P4G
python src/data/download_datasets.py --output-dir data/raw --dataset persuasionforgood

# Preprocess P4G
python src/data/preprocess_persuasionforgood.py \
    --input-dir data/raw/persuasionforgood \
    --output-file data/processed/p4g_examples.jsonl

# Create dataset with both CMV and P4G
python src/data/create_sft_dataset.py \
    --use-cmv \
    --use-p4g \
    --max-examples 30000
```

This will mix CMV and P4G examples up to 30k total.

### CMV Only, No Limit

Use all available CMV data:

```bash
python src/data/create_sft_dataset.py \
    --use-cmv \
    --max-examples 0  # 0 means no limit
```

### P4G Only (Not Recommended)

If you want to use only P4G:

```bash
python src/data/create_sft_dataset.py \
    --no-cmv \
    --use-p4g \
    --max-examples 30000
```

## Expected Results with 30k CMV

### Dataset Composition

- **Training**: 24,000 examples (80%)
- **Validation**: 3,000 examples (10%)
- **Test**: 3,000 examples (10%)

### Training Time

- **g5.xlarge**: ~6-8 hours
- **g4dn.xlarge**: ~9-12 hours

### Cost

- **g5.xlarge**: ~$6-8
- **g4dn.xlarge**: ~$4.50-6

### Model Quality

- Should generate coherent persuasive arguments
- May not be as polished as 50k model
- Still sufficient for RL improvements via DPO
- Good balance of cost vs. performance

## Comparison: 30k vs 50k Examples

| Metric                   | 30k Examples                       | 50k Examples           |
| ------------------------ | ---------------------------------- | ---------------------- |
| Training Time            | 6-8 hours                          | 10-12 hours            |
| Cost (g5.xlarge)         | ~$6-8                              | ~$10-12                |
| SFT Quality              | Good                               | Better                 |
| RL Improvement Potential | Higher                             | Moderate               |
| Recommended For          | Budget-conscious, faster iteration | Best final performance |

### When to Use 30k:

- ✅ Limited budget (<$10)
- ✅ Want faster iterations
- ✅ Time constraints (need results in <8 hours)
- ✅ CS230 project with limited resources
- ✅ Want more room for RL improvements

### When to Use 50k:

- ✅ Have $15-20 budget
- ✅ Want strongest possible SFT base
- ✅ Publishing/competitive results
- ✅ Can afford 10-12 hour training time

## Verifying Configuration

Check your current settings:

```bash
# View data config
cat configs/data_config.yaml | grep -A 3 "datasets:"

# Check what will be used
python src/data/create_sft_dataset.py --help | grep -A 5 "use-cmv"
```

Expected output:

```
datasets:
  use_cmv: true
  use_p4g: false
  max_examples: 30000
```

## Troubleshooting

### "No examples loaded"

**Problem**: Script says "No examples loaded"

**Solution**: Make sure CMV is preprocessed:

```bash
python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl
```

### "CMV file not found"

**Problem**: Can't find CMV examples

**Solution**: Download and preprocess first:

```bash
python src/data/download_datasets.py --dataset cmv
python src/data/preprocess_cmv.py
```

### "Only X examples available"

**Problem**: Fewer than 30k CMV examples found

**Solutions**:

1. **Loosen filters**: Use wider token range in `preprocess_cmv.py`:
   ```bash
   python src/data/preprocess_cmv.py --min-tokens 15 --max-tokens 1500
   ```
2. **Use what you have**: Let it use all available:
   ```bash
   python src/data/create_sft_dataset.py --max-examples 0
   ```

## FAQ

### Q: Is 30k enough for good SFT performance?

**A**: Yes, 30k is the minimum recommended size. It's 100x your RLHF data (300) and 3x your RLAIF data (10k), which matches successful RL-from-HF papers. The model will learn persuasion patterns well enough for DPO to improve it.

### Q: Why not use PersuasionForGood?

**A**: Based on your assessment, P4G quality was lower than desired. CMV has verified persuasive success (delta awards) and more natural language. You can always re-enable P4G later if needed.

### Q: Can I train with fewer examples (e.g., 10k)?

**A**: Not recommended. With 10k SFT examples, your model barely learns the task, leaving too much for RL to fix. 30k is the practical minimum for your RL plan (300 RLHF, 10k RLAIF).

### Q: Should I use more than 30k?

**A**: If budget allows:

- **40k**: Sweet spot for stronger SFT without much extra cost (+$2-3)
- **50k**: Recommended optimal size (+$4-5)
- **60k+**: Diminishing returns, only if data is abundant

### Q: How do I switch back to P4G?

**A**: Edit `configs/data_config.yaml`:

```yaml
datasets:
  use_p4g: true # Change to true
```

Or use command-line flag:

```bash
python src/data/create_sft_dataset.py --use-p4g
```

### Q: Does this affect preference data generation?

**A**: No. Preference generation uses the trained SFT model, not the raw datasets. It works the same regardless of whether you trained on CMV, P4G, or both.

## Summary

**New Defaults**:

- ✅ 100% CMV dataset (no P4G)
- ✅ 30,000 examples
- ✅ Same quality, lower cost
- ✅ Faster training (~6-8 hours)
- ✅ Budget-friendly (~$6-8)

**P4G code is preserved** but disabled by default. You can re-enable it anytime with `--use-p4g` flag.

This configuration balances quality, cost, and training time while providing sufficient data for effective RL fine-tuning.
