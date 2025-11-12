# Final Evaluation Guide

## Overview

After training all your models (Base, SFT, RLHF-only, RLAIF→RLHF), you'll need to evaluate them on a **completely held-out test set** for unbiased comparison.

## Held-Out Test Set

During dataset creation (Step 4), the system automatically reserves **115 examples** that are:

✅ **Never used for SFT training** (reserved before any splits)  
✅ **Never used for preference generation** (not in AI or Human pools)  
✅ **Never seen during any RL training**  
✅ **Perfect for final evaluation**

**Location**: `data/processed/final_eval_reserved.jsonl`

## Dataset Breakdown

```
Total CMV data: 11,865 examples
├─ Final Eval: 115 → RESERVED FIRST (held-out) ⭐ 🔒
└─ Remaining: 11,750 → Split into:
    ├─ Training: 9,400 (80%) → Used for SFT training
    └─ Val + Test: 2,350 (20%) → Shuffled together, then split:
        ├─ 2,150 (91.5%) → Used for RLAIF preference generation
        └─ 200 (8.5%) → Used for RLHF preference generation
```

**Key Point**: The 115 examples are set aside FIRST in Step 4 (`create_sft_dataset.py`), before any training splits are created. This ensures they are completely held-out from all training and preference generation.

## Why This Matters

### Without Held-Out Set ❌

- Evaluate on same data used for preferences → Biased results
- Can't tell if improvements are real or just overfitting
- Results not comparable across models

### With Held-Out Set ✅

- All models evaluated on identical, unseen data
- Fair comparison without any training bias
- Results are scientifically valid for your report
- Standard practice in ML research

## Evaluation Procedure

### Step 1: Generate Responses from All Models

After training all models, generate responses on the held-out set:

```bash
# Activate environment
source venv/bin/activate

# Evaluate Base model (no fine-tuning)
python src/eval/evaluate_model.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/final_eval_reserved.jsonl \
    --num-samples 115 \
    --output-file results/base_final_eval.json

# Evaluate SFT model
python src/eval/evaluate_model.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/final_eval_reserved.jsonl \
    --num-samples 115 \
    --output-file results/sft_final_eval.json

# Evaluate RLHF-only model
python src/eval/evaluate_model.py \
    --model-path models/checkpoints/qwen-rlhf/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/final_eval_reserved.jsonl \
    --num-samples 115 \
    --output-file results/rlhf_final_eval.json

# Evaluate RLAIF→RLHF model
python src/eval/evaluate_model.py \
    --model-path models/checkpoints/qwen-rlaif-rlhf/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/final_eval_reserved.jsonl \
    --num-samples 115 \
    --output-file results/rlaif_rlhf_final_eval.json
```

### Step 2: Use Grader Model to Rank Responses

For each held-out prompt, you'll have 4 responses (one from each model). Use your grader model to rank them:

```python
# Example grader prompt structure
grader_prompt = """
You will see a conversation context and four candidate replies (A, B, C, D).
Rank the replies from MOST PERSUASIVE to LEAST PERSUASIVE, considering:
- Clarity and coherence
- Respectful, non-manipulative tone
- Direct engagement with concerns
- Reasonable evidence or arguments

Answer with exactly one line:
RANKING: A > C > B > D

[CONTEXT]
{context}

[CANDIDATE A] {response_A}
[CANDIDATE B] {response_B}
[CANDIDATE C] {response_C}
[CANDIDATE D] {response_D}
"""
```

### Step 3: Calculate Scores

Assign points based on rankings:

- 1st place: 3 points
- 2nd place: 2 points
- 3rd place: 1 point
- 4th place: 0 points

Aggregate across all 115 prompts:

```
Score(model) = Total Points / (3 × 115)
```

This normalizes scores to 0-1 range (1.0 = always ranked first).

## Expected Results

Based on the project design, you should see something like:

| Model      | Expected Score | Interpretation                        |
| ---------- | -------------- | ------------------------------------- |
| Base       | 0.15-0.25      | Weakest (no fine-tuning)              |
| SFT        | 0.35-0.45      | Strong baseline (supervised learning) |
| RLHF-only  | 0.40-0.50      | Improved by human preferences         |
| RLAIF→RLHF | 0.45-0.55      | Best (AI warmup + human alignment)    |

**Key Questions for Your Report**:

1. Does RLHF-only beat SFT? (Human preferences help?)
2. Does RLAIF→RLHF beat RLHF-only? (AI warmup helps?)
3. By how much? (Effect sizes matter!)

## Statistical Analysis

With 115 test prompts, you have reasonable statistical power for detecting meaningful differences:

### Win Rate Analysis

For each pair of models, count how often model A beats model B:

```python
wins_A = sum(rank[A] > rank[B] for prompt in prompts)
win_rate_A = wins_A / 115
```

### Confidence Intervals

With n=115, you can compute 95% confidence intervals:

```python
import scipy.stats as stats

# For win rate
se = sqrt(p * (1-p) / n)
ci_lower = p - 1.96 * se
ci_upper = p + 1.96 * se
```

### Significance Testing

Use paired t-test (since same prompts for all models):

```python
from scipy.stats import ttest_rel

scores_sft = [...]  # Score for each prompt
scores_rlhf = [...]  # Score for each prompt

t_stat, p_value = ttest_rel(scores_rlhf, scores_sft)
print(f"RLHF vs SFT: p={p_value:.4f}")
```

## Qualitative Analysis

Beyond numbers, analyze specific examples:

### 1. Where RLHF Beats SFT

Find prompts where RLHF >> SFT:

- What made RLHF better?
- More respectful? Better arguments?
- Less manipulative?

### 2. Where RLAIF→RLHF Beats RLHF-only

Find prompts where RLAIF warmup helped:

- Did AI preferences provide good foundation?
- Did human preferences refine it?

### 3. Failure Cases

Find prompts where all models failed:

- What's hard about these prompts?
- Are they edge cases?
- Do they reveal model limitations?

## Report Structure

### For CS230 Final Report

**Evaluation Section** should include:

1. **Dataset**:

   - "115 held-out examples, reserved before any training splits, never used for training or preference generation"
   - Explain why this ensures unbiased comparison

2. **Grader Model**:

   - Which model used (GPT-4, Claude, etc.)
   - Grading criteria (persuasiveness, respect, etc.)
   - Position bias control (randomize A/B/C/D)

3. **Quantitative Results**:

   - Table with scores for all 4 models
   - Win rates for pairwise comparisons
   - Statistical significance tests
   - Confidence intervals

4. **Qualitative Results**:

   - 3-5 example prompts showing interesting patterns
   - Where each model excels
   - Common failure modes

5. **Analysis**:
   - Does RLHF help? How much?
   - Does RLAIF warmup help? How much?
   - Is it worth the extra complexity?

## Common Pitfalls to Avoid

❌ **Don't** evaluate on training data  
❌ **Don't** use preference pool prompts (contaminated)  
❌ **Don't** cherry-pick examples  
❌ **Don't** ignore statistical significance  
❌ **Don't** forget to randomize grader position (A/B/C/D)

✅ **Do** use the held-out set (final_eval_reserved.jsonl)  
✅ **Do** report all results (wins and losses)  
✅ **Do** include error bars/confidence intervals  
✅ **Do** show example outputs  
✅ **Do** discuss limitations

## Tips for Strong Results

1. **Consistent Evaluation**: Use same grader for all models
2. **Multiple Seeds**: If time allows, generate multiple responses and average
3. **Position Bias**: Randomize which model is A/B/C/D
4. **Ablation Studies**: Test impact of each component
5. **Human Validation**: Manually check a subset (~50 prompts)

## File Organization

```
results/
├── base_final_eval.json          # Base model responses
├── sft_final_eval.json            # SFT model responses
├── rlhf_final_eval.json           # RLHF-only responses
├── rlaif_rlhf_final_eval.json     # RLAIF→RLHF responses
├── grader_rankings.jsonl          # Rankings from grader model
├── scores_summary.json            # Aggregated scores
└── analysis_report.md             # Qualitative analysis
```

## Next Steps After SFT Phase

After SFT completes and you have the held-out set saved:

1. ✅ Train RLAIF model (DPO on AI preferences)
2. ✅ Train RLHF-only model (DPO on human preferences)
3. ✅ Train RLAIF→RLHF model (two-stage DPO)
4. ✅ Generate responses from all 4 models on held-out set
5. ✅ Use grader model to rank responses
6. ✅ Calculate scores and statistical tests
7. ✅ Write report with findings

---

**The held-out evaluation set is your key to credible, publishable results!** 🎯
