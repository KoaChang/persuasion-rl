# SFT Training Fix: Proper Label Masking

## The Problem

The original SFT training code was supervising **all tokens**, including:
- System prompts
- Context/conversation history
- Task instructions
- The actual response

This meant the model was learning to:
1. ❌ Memorize and reproduce system prompts
2. ❌ Regenerate conversation context
3. ❌ Repeat task instructions
4. ✅ Generate persuasive responses (only 1 out of 4!)

## Why This Was Wrong

Your goal is to make the model good at **generating persuasive responses** given a context. The model should learn:

**"Given this context → generate this type of response"**

NOT:

**"Memorize this entire template and conversation"**

## The Fix

### What Changed

In `src/sft/train_sft.py`, the `preprocess_function` now:

1. **Tokenizes the prompt separately** to find its length
2. **Masks prompt tokens with -100** (PyTorch's CrossEntropyLoss ignores these)
3. **Only supervises response tokens** (the persuasive comments)
4. **Also masks padding tokens** with -100

### Code Changes

**Before (WRONG):**
```python
def preprocess_function(examples, tokenizer, max_length):
    result = tokenizer(examples["full_text"], ...)
    result["labels"] = result["input_ids"].copy()  # ❌ Supervises everything!
    return result
```

**After (CORRECT):**
```python
def preprocess_function(examples, tokenizer, max_length):
    # Find prompt length
    prompt_only = tokenizer(examples["input_text"], ...)
    prompt_length = len(prompt_only["input_ids"])
    
    # Tokenize full text
    result = tokenizer(examples["full_text"], ...)
    
    # Mask prompt tokens with -100
    labels = result["input_ids"].copy()
    for i in range(prompt_length):
        labels[i] = -100  # ✅ Don't supervise prompts!
    
    # Mask padding too
    for i in range(len(labels)):
        if labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    
    result["labels"] = labels
    return result
```

## How Label Masking Works

In PyTorch's `CrossEntropyLoss` (used by causal language models):
- Tokens with label `-100` are **ignored** in loss computation
- Only tokens with actual IDs contribute to the gradient

**Example:**

```
Input sequence:  [SYSTEM] You are... [CONTEXT] ... [TASK] ... [RESPONSE] This is persuasive
                 |________Prompt (masked)_________|  |____Response (supervised)____|
Labels:          [-100, -100, -100, ..., -100, 1234, 5678, 9012, ...]
                  ^                        ^      ^                   ^
                  |                        |      |                   |
              Ignored in loss          Last prompt  Supervised tokens |
                                         token                        |
```

## Impact on Training

### Before Fix (Wrong Objective)
- **Loss computed on**: System prompt (30%) + Context (40%) + Task (10%) + Response (20%)
- **Model learns**: Template memorization + some response generation
- **Effective training**: Only ~20% focused on your actual goal
- **Risk**: Model overfits to exact prompt format, doesn't generalize

### After Fix (Correct Objective)  
- **Loss computed on**: Response (100%)
- **Model learns**: How to generate persuasive responses given context
- **Effective training**: 100% focused on persuasion skill
- **Benefit**: Model learns the actual task, generalizes better

## Verification

Run the verification script to see the fix in action:

```bash
source venv/bin/activate
python tests/verify_label_masking.py
```

This will show you:
- How many tokens are masked (prompt)
- How many tokens are supervised (response)
- What the model is actually being trained on

## Why Both `input_text` and `full_text` Are Needed

The dataset now includes:
- `input_text`: Prompt without response (used to calculate prompt length)
- `full_text`: Prompt + response (the complete training sequence)

We need both because:
1. We tokenize `input_text` to find where the prompt ends
2. We tokenize `full_text` to get the complete sequence
3. We mask everything from 0 to `len(prompt)` with -100
4. Loss is only computed on the response portion

## Re-training Recommendation

If you've already trained a model with the old (buggy) approach, you should **retrain from scratch** with this fix. The old model learned the wrong objective and won't be as good at the actual persuasion task.

## Summary

✅ **Now your model will learn**: "Given conversation context, generate a persuasive response"  
❌ **Not**: "Memorize this entire prompt template"

This is the standard practice for instruction fine-tuning and will result in a much better persuasion model!

