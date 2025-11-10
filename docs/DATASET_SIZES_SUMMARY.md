# Dataset Sizes Summary

## Updated Configuration (50k SFT)

### Overview

After analysis of available data and RL requirements, the default configuration has been updated to:

**SFT Training**: 50,000 examples from CMV  
**RLAIF Pool**: 8,000 prompts (from val + test)  
**RLHF Pool**: 300 prompts (from test)

## Breakdown

### SFT Dataset (50k total)

| Split     | Examples | Percentage | Purpose        |
| --------- | -------- | ---------- | -------------- |
| **Train** | 40,000   | 80%        | SFT training   |
| **Val**   | 5,000    | 10%        | SFT validation |
| **Test**  | 5,000    | 10%        | SFT testing    |

**Source**: 100% CMV (ChangeMyView) delta-winning comments

### Preference Generation

| Pool          | Size  | Source                  | Purpose                          |
| ------------- | ----- | ----------------------- | -------------------------------- |
| **RLAIF**     | 8,000 | Val (5k) + Test (3k)    | AI-graded preferences for DPO    |
| **RLHF**      | 300   | Test (~300)             | Human-graded preferences for DPO |
| **Final Eval**| ~1,700| Val + Test (remaining)  | Held-out for unbiased evaluation ⭐ |

**Note**: Each prompt gets 2 responses from SFT model (different random seeds)

**⭐ Final Eval Set**: These ~1,700 prompts are NEVER used for preference generation or training - they're completely held-out for unbiased final evaluation of all models.

## Key Ratios

### SFT : RL Ratios

| Comparison   | Ratio                | Assessment                        |
| ------------ | -------------------- | --------------------------------- |
| SFT : RLAIF  | 50k : 8k = **6.25x** | ✅ Optimal (5-10x is ideal)       |
| SFT : RLHF   | 50k : 300 = **166x** | ✅ Excellent (100-200x is ideal)  |
| RLAIF : RLHF | 8k : 300 = **26x**   | ✅ Good (20-50x allows AI warmup) |

### Comparison to Literature

| System           | SFT Size | RLAIF Size | RLHF Size | SFT:RLAIF | SFT:RLHF |
| ---------------- | -------- | ---------- | --------- | --------- | -------- |
| **Your Config**  | 50k      | 8k         | 300       | 6.25x     | 166x     |
| Anthropic        | 50k+     | 5-10k      | 1-5k      | 5-10x     | 10-50x   |
| InstructGPT      | 13k      | -          | 33k       | -         | 0.4x     |
| Typical Research | 30-100k  | 3-20k      | 300-5k    | 5-15x     | 50-200x  |

**Your ratios are well within best practices!** ✅

## Why This Configuration?

### Why 50k SFT? (up from 30k)

**Pros**:

- ✅ Stronger baseline model → better RL starting point
- ✅ More diverse persuasion patterns
- ✅ Provides enough val + test data (10k) for preferences
- ✅ Still cost-effective (~$10-12 vs ~$6-8 for 30k)

**Cons**:

- ⚠️ ~40% more training time (10-12 hours vs 6-8 hours)
- ⚠️ ~60% higher cost ($10-12 vs $6-8)

**Decision**: Worth it for higher quality and sufficient preference data

### Why 8k RLAIF? (down from 10k)

**Pros**:

- ✅ No duplicates (exactly fits val + test sets)
- ✅ Cheaper AI labeling (~$80-100 vs ~$100-150)
- ✅ Still 26x your human data (excellent ratio)
- ✅ 6.25x SFT size (optimal range)

**Cons**:

- ⚠️ Slightly less AI preference data than originally planned

**Decision**: Better to avoid duplicates, still plenty of data

### Why 300 RLHF? (unchanged)

**Pros**:

- ✅ Manageable for manual labeling (~3-5 hours of work)
- ✅ 166x ratio to SFT (ideal for fine-tuning)
- ✅ Standard size in research (200-500 typical)

**Cons**:

- ⚠️ Small dataset → need careful DPO hyperparameters

**Decision**: Perfect for CS230 project scope

## Data Flow

```
CMV Dataset (delta-winning comments)
    ↓
Preprocessing & Filtering
    ↓
50,000 examples
    ↓
    ├── 40,000 (80%) → SFT Training
    ├── 5,000 (10%) → SFT Validation
    └── 5,000 (10%) → SFT Testing
            ↓
    Val + Test shuffled together (10,000 prompts)
            ↓
            ├── 300 prompts → RLHF pool
            ├── 8,000 prompts → RLAIF pool
            └── ~1,700 prompts → Final Eval (held-out) ⭐
```

**⭐ Key Feature**: The ~1,700 held-out prompts ensure unbiased final evaluation across all models!

## Cost Breakdown

### Training Costs (g5.xlarge @ $1.01/hour)

| Phase                 | Time            | Cost       |
| --------------------- | --------------- | ---------- |
| SFT Training (50k)    | 10-12 hours     | $10-12     |
| Preference Generation | 3-4 hours       | $3-4       |
| **Total SFT Phase**   | **13-16 hours** | **$13-16** |

### Labeling Costs (estimated)

| Phase              | Amount      | Cost per Label | Total       |
| ------------------ | ----------- | -------------- | ----------- |
| RLAIF (GPT-4 API)  | 8,000 pairs | $0.01-0.015    | $80-120     |
| RLHF (Human time)  | 300 pairs   | $0 (DIY)       | $0          |
| **Total Labeling** |             |                | **$80-120** |

### DPO Training Costs (future)

| Phase          | Time          | Cost     |
| -------------- | ------------- | -------- |
| RLAIF DPO      | 3-5 hours     | $3-5     |
| RLHF DPO       | 1-2 hours     | $1-2     |
| RLAIF→RLHF DPO | 1-2 hours     | $1-2     |
| **Total DPO**  | **5-9 hours** | **$5-9** |

### Total Project Cost

**SFT Phase**: $13-16  
**Labeling**: $80-120 (mostly GPT-4 API)  
**DPO Phase**: $5-9  
**Total**: **$98-145**

For a CS230 project, this is very reasonable!

## Available Data Check

### With 50k Total Examples

From test + val sets:

- Val: 5,000 prompts
- Test: 5,000 prompts
- **Total available**: 10,000 prompts ✅

Allocation:

- RLAIF needs: 8,000 prompts ✅
- RLHF needs: 300 prompts ✅
- **Total needed**: 8,300 prompts
- **Remaining**: 1,700 prompts (buffer)

**No sampling with replacement needed!** ✅

## Alternative Configurations

### Budget Option: 30k SFT

```
SFT: 30k (24k train / 3k val / 3k test)
RLAIF: 5k (from val + test)
RLHF: 300 (from test)
Cost: ~$8 SFT + $50-75 RLAIF + $3-5 DPO = $61-88 total
Time: ~9-12 hours
```

**Tradeoffs**: Cheaper, faster, but weaker baseline and less RLAIF data

### Maximum Quality: 100k SFT

```
SFT: 100k (80k train / 10k val / 10k test)
RLAIF: 9.7k (from val + test)
RLHF: 300 (from test)
Cost: ~$20 SFT + $100-150 RLAIF + $5-9 DPO = $125-179 total
Time: ~25-30 hours
```

**Tradeoffs**: Best quality, but expensive and slower. Diminishing returns.

## Recommendation

**Stick with 50k configuration** ✅

**Rationale**:

- Strong SFT baseline without excessive cost
- Optimal ratios for both RLAIF and RLHF
- No data duplication issues
- Well within best practices from research
- Reasonable budget for CS230 (~$100-145 total)

## Verification

After running data preprocessing, verify sizes:

```bash
# Check SFT splits
wc -l data/processed/sft_train.jsonl  # Should be ~40,000
wc -l data/processed/sft_val.jsonl    # Should be ~5,000
wc -l data/processed/sft_test.jsonl   # Should be ~5,000

# Check preference data
wc -l data/preferences/ai_pool_responses.jsonl     # Should be 8,000
wc -l data/preferences/human_pool_responses.jsonl  # Should be 300
```

## Next Steps

After SFT completes:

1. ✅ You'll have 50k trained examples
2. ✅ Generate 8k RLAIF pairs + 300 RLHF pairs
3. 🔄 Label RLAIF with GPT-4 (~$80-120, ~2-4 hours)
4. 🔄 Label RLHF manually (~3-5 hours of your time)
5. 🔄 Train DPO models (RLAIF-only, RLHF-only, RLAIF→RLHF)
6. 🔄 Evaluate all models and compare

---

**Configuration last updated**: Based on user request to increase SFT to 50k and RLAIF to 8k
