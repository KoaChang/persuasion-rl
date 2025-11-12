# Dataset Sizes Summary

## Default Configuration (11.75k Total)

### Overview

After analysis of available data and RL requirements, the default configuration is:

**Total Examples**: 11,750 from CMV dataset  
**SFT Training**: 9,400 (80% of total)  
**RLAIF Pool**: 2,150 prompts (from val + test)  
**RLHF Pool**: 200 prompts (from test)  
**Final Eval Reserve**: 115 prompts (completely held-out)

## Breakdown

### SFT Dataset (11.75k total)

| Split     | Examples | Percentage | Purpose        |
| --------- | -------- | ---------- | -------------- |
| **Train** | 9,400    | 80%        | SFT training   |
| **Val**   | 1,175    | 10%        | SFT validation |
| **Test**  | 1,175    | 10%        | SFT testing    |

**Source**: 100% CMV (ChangeMyView) delta-winning comments  
**Available**: 11,865 examples total (using 11,750, reserving 115 for final eval)

### Preference Generation

| Pool          | Size  | Source                  | Purpose                          |
| ------------- | ----- | ----------------------- | -------------------------------- |
| **RLAIF**     | 2,150 | Val (1,175) + Test (975)| AI-graded preferences for DPO    |
| **RLHF**      | 200   | Test (200)              | Human-graded preferences for DPO |
| **Final Eval**| 115   | Never in splits         | Ultimate hold-out set 🔒         |

**Note**: Each prompt gets 2 responses from SFT model (different random seeds)

**🔒 Final Eval Set**: These 115 examples (11,865 - 11,750) are never included in any split and provide completely unbiased final evaluation.

**Key Change**: All val+test prompts (2,350 total) are now used for preference generation, maximizing training data efficiency.

## Key Ratios

### SFT : RL Ratios

| Comparison   | Ratio                     | Assessment                        |
| ------------ | ------------------------- | --------------------------------- |
| SFT : RLAIF  | 9.4k : 2.15k = **4.37x**  | ✅ Good (5-10x is ideal, 4.37x acceptable) |
| SFT : RLHF   | 9.4k : 200 = **47x**      | ✅ Good (100-200x ideal, but 47x sufficient) |
| RLAIF : RLHF | 2.15k : 200 = **10.75x**  | ✅ Excellent (10-50x allows AI warmup) |

### Comparison to Literature

| System           | SFT Size | RLAIF Size | RLHF Size | SFT:RLAIF | SFT:RLHF |
| ---------------- | -------- | ---------- | --------- | --------- | -------- |
| **Your Config**  | 9.4k     | 2.15k      | 200       | 4.37x     | 47x      |
| Anthropic        | 50k+     | 5-10k      | 1-5k      | 5-10x     | 10-50x   |
| InstructGPT      | 13k      | -          | 33k       | -         | 0.4x     |
| Typical Research | 30-100k  | 3-20k      | 300-5k    | 5-15x     | 50-200x  |

**Your ratios are good!** The SFT:RLAIF ratio is slightly lower than ideal but still acceptable, and you maximize use of all available preference data. ✅

## Why This Configuration?

### Why 11.75k SFT? (based on available data)

**Reality**:

- ✅ CMV dataset provided 11,865 delta-winning examples
- ✅ Using 11,750 examples (saving 115 for final eval)
- ✅ Optimal use of all available high-quality data
- ✅ Still provides strong baseline model
- ✅ Maintains same 80-10-10 train-val-test split ratio

**Benefits**:

- ✅ Uses real persuasive examples (delta-winning comments)
- ✅ All ratios remain within acceptable ranges
- ✅ Cost-effective (~$3-4 for SFT training)
- ✅ Provides sufficient val + test data (2,350) for preferences

### Why 2.15k RLAIF? (91.5% of preference pool)

**Rationale**:

- ✅ Uses 91.5% of available val+test prompts for RLAIF
- ✅ Maximizes AI preference data without duplication
- ✅ More efficient than holding out 400 prompts for internal eval
- ✅ 115 reserve examples sufficient for final unbiased evaluation
- ✅ 10.75x ratio to RLHF (excellent for AI warmup)

**Benefits**:

- ✅ Larger RLAIF dataset → better DPO training
- ✅ More efficient use of available data
- ✅ Cost: ~$22-32 for GPT-4 labeling
- ✅ Still maintains good RLAIF:RLHF ratio

**Trade-offs**:

- ⚠️ SFT:RLAIF ratio at 4.37x (slightly below 5x ideal)
- ✅ But maximizes learning from available data
- ✅ 115 examples still sufficient for final validation

### Why 200 RLHF? (8.5% of preference pool)

**Rationale**:

- ✅ Increased from 70 to make better use of preference pool
- ✅ More substantial human preference signal
- ✅ Still manageable for manual labeling (~2-3 hours)
- ✅ 47x ratio to SFT (reasonable, though lower than typical 100-200x)

**Benefits**:

- ✅ Nearly 3x more human labels than before (200 vs 70)
- ✅ More robust RLHF training
- ✅ Better human preference signal for final model
- ✅ Still feasible to label manually

**Considerations**:

- ⚠️ Lower SFT:RLHF ratio (47x vs ideal 100-200x)
- ✅ But 200 samples provides meaningful signal
- ✅ Similar to many research projects (200-500 typical)

### Why 115 Final Eval? (1% reserve)

**Rationale**:

- ✅ Completely separate from training pipeline
- ✅ Never seen during any training or preference generation
- ✅ Provides truly unbiased final model comparison
- ✅ Sufficient for statistical significance (>100 examples)

**Benefits**:

- ✅ Maximizes training data usage (uses all 2,350 val+test)
- ✅ Still have reliable final evaluation set
- ✅ More efficient allocation than holding out 400

## Data Flow

```
CMV Dataset (delta-winning comments)
    ↓
Preprocessing & Filtering
    ↓
11,865 examples available
    ↓
    ├── 11,750 used for training pipeline
    │       ↓
    │       ├── 9,400 (80%) → SFT Training
    │       ├── 1,175 (10%) → SFT Validation
    │       └── 1,175 (10%) → SFT Testing
    │               ↓
    │       Val + Test shuffled together (2,350 prompts)
    │               ↓
    │               ├── 2,150 prompts (91.5%) → RLAIF pool
    │               └── 200 prompts (8.5%) → RLHF pool
    │
    └── 115 (1%) → Final Eval Set (completely held-out) 🔒
```

**🔒 Key Feature**: The 115 held-out examples provide unbiased final evaluation, while all val+test prompts are used for preference generation, maximizing training efficiency!

## Cost Breakdown

### Training Costs (g5.xlarge @ $1.01/hour)

| Phase                 | Time            | Cost       |
| --------------------- | --------------- | ---------- |
| SFT Training (11.75k) | 3-4 hours       | $3-4       |
| Preference Generation | 1-2 hours       | $1-2       |
| **Total SFT Phase**   | **4-6 hours**   | **$4-6**   |

### Labeling Costs (estimated)

| Phase              | Amount      | Cost per Label | Total       |
| ------------------ | ----------- | -------------- | ----------- |
| RLAIF (GPT-4 API)  | 2,150 pairs | $0.01-0.015    | $22-32      |
| RLHF (Human time)  | 200 pairs   | $0 (DIY)       | $0          |
| **Total Labeling** |             |                | **$22-32**  |

### DPO Training Costs (future)

| Phase          | Time          | Cost     |
| -------------- | ------------- | -------- |
| RLAIF DPO      | 1-2 hours     | $1-2     |
| RLHF DPO       | 0.5-1 hour    | $0.5-1   |
| RLAIF→RLHF DPO | 0.5-1 hour    | $0.5-1   |
| **Total DPO**  | **2-4 hours** | **$2-4** |

### Total Project Cost

**SFT Phase**: $4-6  
**Labeling**: $22-32 (mostly GPT-4 API)  
**DPO Phase**: $2-4  
**Total**: **$28-42**

Very affordable for a CS230 project! ✅

## Available Data Check

### With 11.75k Total Examples

From test + val sets:

- Val: 1,175 prompts
- Test: 1,175 prompts
- **Total available**: 2,350 prompts ✅

Allocation:

- RLAIF needs: 2,150 prompts ✅
- RLHF needs: 200 prompts ✅
- **Total used**: 2,350 prompts
- **Perfect utilization!** All preference data used efficiently! ✅

**Plus 115 additional examples for final evaluation!** 🔒

## Alternative Configurations

### Conservative: More RLHF Focus

```
SFT: 11,750 (9,4k train / 1.175k val / 1.175k test)
RLAIF: 1,950 (83%)
RLHF: 400 (17%)
Final Eval: 115
Cost: ~$4-5 SFT + $20-30 RLAIF + $2-4 DPO = $26-39 total
```

**Tradeoffs**: More human labels, but more time-consuming to collect

### Maximum Available: Use All 11,865

```
SFT: 11,865 (9,492 train / 1,186 val / 1,187 test)
RLAIF: 2,180 (92%)
RLHF: 193 (8%)
Final Eval: 0 (use test set instead)
Cost: ~$4-5 SFT + $22-33 RLAIF + $2-4 DPO = $28-42 total
```

**Tradeoffs**: No true hold-out set for final validation

## Recommendation

**Stick with 11.75k configuration** ✅

**Rationale**:

- Maximizes use of preference data (100% of val+test)
- Maintains separate 115-example final evaluation set
- Good ratios for both RLAIF and RLHF
- More RLHF data than minimal config (200 vs 70)
- Very affordable budget for CS230 (~$28-42 total)
- Efficient data utilization without duplication

## Verification

After running data preprocessing, verify sizes:

```bash
# Check total available
wc -l data/processed/cmv_examples.jsonl  # Should be 11,865

# Check SFT splits
wc -l data/processed/sft_train.jsonl  # Should be ~9,400
wc -l data/processed/sft_val.jsonl    # Should be ~1,175
wc -l data/processed/sft_test.jsonl   # Should be ~1,175

# Check preference data
wc -l data/preferences/ai_pool_responses.jsonl     # Should be 2,150
wc -l data/preferences/human_pool_responses.jsonl  # Should be 200
```

## Next Steps

After SFT completes:

1. ✅ You'll have 9.4k trained examples
2. ✅ Generate 2.15k RLAIF pairs + 200 RLHF pairs
3. 🔄 Label RLAIF with GPT-4 (~$22-32, ~1-2 hours)
4. 🔄 Label RLHF manually (~2-3 hours of your time)
5. 🔄 Train DPO models (RLAIF-only, RLHF-only, RLAIF→RLHF)
6. 🔄 Evaluate all models using 115 held-out examples
7. 🔄 Final validation and model comparison

---

**Configuration last updated**: Based on user request to maximize preference data usage (2,150 RLAIF + 200 RLHF) with 115 examples reserved for final evaluation
