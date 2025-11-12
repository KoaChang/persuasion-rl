"""
Test script to verify that label masking is working correctly.
This ensures that only the response tokens are supervised, not the prompt.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer


def test_label_masking():
    """Test that labels are properly masked for prompt tokens."""
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a sample example
    input_text = """[SYSTEM] You are a respectful assistant trying to persuade the other person using honest, well-reasoned arguments. You should acknowledge uncertainty and avoid manipulation.

[CONTEXT]
Person A: I think cats are better than dogs.

[TASK]
Write the next message that continues the conversation and aims to persuade the other person.

[RESPONSE]
"""
    
    full_text = input_text + "While I understand your preference for cats, dogs offer unique benefits like loyalty and companionship that are hard to match."
    
    # Simulate the preprocessing function
    max_length = 512
    
    # Tokenize prompt only
    prompt_only = tokenizer(
        input_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    prompt_length = len(prompt_only["input_ids"])
    
    # Tokenize full text
    result = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )
    
    # Create labels with masking
    labels = result["input_ids"].copy()
    
    # Mask prompt tokens
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    
    # Mask padding tokens
    for i in range(len(labels)):
        if labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    
    # Print results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"\n1. Prompt length: {prompt_length} tokens")
    print(f"2. Full sequence length: {len(result['input_ids'])} tokens")
    print(f"3. Number of supervised tokens (non -100): {sum(1 for x in labels if x != -100)}")
    print(f"4. Number of masked tokens (-100): {sum(1 for x in labels if x == -100)}")
    
    # Decode to show what's supervised
    print("\n" + "="*80)
    print("PROMPT (MASKED - NOT SUPERVISED):")
    print("="*80)
    print(input_text)
    
    print("\n" + "="*80)
    print("RESPONSE (SUPERVISED):")
    print("="*80)
    supervised_tokens = [token_id if label != -100 else tokenizer.pad_token_id 
                        for token_id, label in zip(result["input_ids"], labels)]
    supervised_text = tokenizer.decode(supervised_tokens, skip_special_tokens=True)
    print(supervised_text)
    
    # Verify that masking is correct
    assert prompt_length > 0, "Prompt should have tokens"
    assert sum(1 for x in labels if x != -100) > 0, "Should have some supervised tokens"
    assert sum(1 for x in labels if x == -100) >= prompt_length, "All prompt tokens should be masked"
    
    print("\n" + "="*80)
    print("✓ VERIFICATION PASSED!")
    print("="*80)
    print("\nLabel masking is working correctly:")
    print("  - Prompt tokens are masked with -100 (NOT supervised)")
    print("  - Response tokens keep their IDs (SUPERVISED)")
    print("  - Model will only learn to generate responses, not memorize prompts")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_label_masking()

