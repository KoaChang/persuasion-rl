# Data Split Changes - Final Eval Reserve

## Summary

Updated the data processing pipeline to properly set aside 115 examples for final model evaluation before creating SFT training splits. This ensures these examples are completely held-out from all training and preference generation.

## Changes Made

### 1. `src/data/create_sft_dataset.py`

**Added:**
- New function `reserve_final_eval_set()` to set aside examples before splitting
- New argument `--reserve-final-eval` (default: 115) to specify number of examples to reserve
- Changed `--max-examples` default from 11750 to None (uses all available after reservation)

**Updated Logic:**
1. Load all examples from CMV dataset (11,865 total)
2. **FIRST**: Reserve 115 examples for final evaluation → saves to `final_eval_reserved.jsonl`
3. **THEN**: Use remaining 11,750 examples for SFT splits (80/10/10)
4. Save train/val/test splits as before

**Output Files:**
- `data/processed/sft_train.jsonl` - 9,400 examples (80%)
- `data/processed/sft_val.jsonl` - 1,175 examples (10%)
- `data/processed/sft_test.jsonl` - 1,175 examples (10%)
- `data/processed/final_eval_reserved.jsonl` - 115 examples (NEW!) 🔒

### 2. `src/sft/generate_preferences.py`

**Updated:**
- Changed `--ai-pool-size` default from 8000 to 2150
- Changed `--human-pool-size` default from 300 to 200
- Updated print messages to clarify that final eval examples were reserved in Step 4

**Behavior:**
- Uses val + test sets (2,350 prompts total)
- Splits into: 2,150 for RLAIF + 200 for RLHF
- No unused prompts (all 2,350 used for preferences)
- The 115 final eval examples are already excluded (reserved in Step 4)

### 3. `scripts/step4_create_sft_dataset.sh`

**Updated:**
- Removed `--max-examples 11750` flag
- Added `--reserve-final-eval 115` flag
- Added comment explaining the reservation

### 4. `scripts/step6_generate_preferences.sh`

**Updated:**
- Added `--val-file data/processed/sft_val.jsonl` to use validation set
- Added `--use-val-set` flag
- Added `--ai-pool-size 2150` to specify RLAIF pool size
- Added `--human-pool-size 200` to specify RLHF pool size

### 5. Documentation Updates

**Updated files:**
- `docs/TODO.md` - Updated pool sizes in step 2
- `docs/DATASET_SIZES_SUMMARY.md` - Updated data flow diagram to show reservation happens first

## Data Flow (New)

```
CMV Dataset (11,865 examples)
    ↓
[Step 4: create_sft_dataset.py]
    ↓
    ├── Reserve 115 → final_eval_reserved.jsonl 🔒
    │   (Never used in training/preferences)
    │
    └── Remaining 11,750 examples
            ↓
            Split 80/10/10:
            ├── 9,400 → sft_train.jsonl
            ├── 1,175 → sft_val.jsonl
            └── 1,175 → sft_test.jsonl
                    ↓
[Step 6: generate_preferences.py]
                    ↓
            Load val + test (2,350 prompts)
                    ↓
                    ├── 2,150 → RLAIF pool
                    └── 200 → RLHF pool
```

## Expected File Sizes

After running the updated pipeline:

| File | Examples | Purpose |
|------|----------|---------|
| `cmv_examples.jsonl` | 11,865 | All preprocessed CMV examples |
| `final_eval_reserved.jsonl` | 115 | Final evaluation (held-out) 🔒 |
| `sft_train.jsonl` | 9,400 | SFT training |
| `sft_val.jsonl` | 1,175 | SFT validation |
| `sft_test.jsonl` | 1,175 | SFT testing |
| `ai_pool_prompts.jsonl` | 2,150 | RLAIF prompts |
| `human_pool_prompts.jsonl` | 200 | RLHF prompts |

**Total:** 11,865 examples
- 9,400 for SFT training
- 2,350 for preference generation (2,150 RLAIF + 200 RLHF)
- 115 held-out for final evaluation 🔒

## Verification

To verify the changes worked correctly, run:

```bash
# Check that final eval set was created
ls -lh data/processed/final_eval_reserved.jsonl

# Verify sizes
wc -l data/processed/cmv_examples.jsonl           # Should be 11,865
wc -l data/processed/final_eval_reserved.jsonl    # Should be 115
wc -l data/processed/sft_train.jsonl              # Should be 9,400
wc -l data/processed/sft_val.jsonl                # Should be 1,175
wc -l data/processed/sft_test.jsonl               # Should be 1,175

# After running step 6:
wc -l data/preferences/ai_pool_prompts.jsonl      # Should be 2,150
wc -l data/preferences/human_pool_prompts.jsonl   # Should be 200
```

## Key Benefits

1. ✅ **Proper Hold-out**: 115 examples are set aside FIRST, ensuring they're never seen during training
2. ✅ **Efficient Data Use**: All 2,350 val+test prompts used for preference generation
3. ✅ **Reproducible**: Random seed ensures same 115 examples reserved every time
4. ✅ **Clear Separation**: Final eval examples in separate file, easy to identify
5. ✅ **Unbiased Evaluation**: Can compare all final models on truly held-out data

## Migration Notes

If you've already run the pipeline:
1. Delete existing splits: `rm data/processed/sft_*.jsonl`
2. Run step 4 again: `bash scripts/step4_create_sft_dataset.sh`
3. This will create new splits + the `final_eval_reserved.jsonl` file
4. Then run step 6 with the updated settings

The random seed (42) ensures reproducibility, so you'll get the same splits each time.

