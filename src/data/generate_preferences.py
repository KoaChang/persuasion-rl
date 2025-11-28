"""
Generate preference data by sampling 2 responses per prompt from the trained SFT model.
Creates AI-graded pool and human-graded pool.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, set_seed

from anthropic import Anthropic

client = Anthropic()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.model_utils import load_trained_model


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_prompts_from_test_set(
    test_file: str,
    ai_pool_size: int,
    human_pool_size: int,
    seed: int = 42,
    val_file: str = None
) -> tuple:
    """
    Extract prompts from test (and optionally val) set for preference generation.
    
    Returns:
        (ai_pool_prompts, human_pool_prompts, unused_prompts)
        unused_prompts: held-out prompts for final evaluation (never used for preferences)
    """
    print(f"Loading test examples from {test_file}...")
    test_data = load_jsonl(test_file)
    
    # We need contexts (prompts without responses) for generation
    prompts = []
    for example in test_data:
        prompts.append({
            "prompt": example["input_text"],
            "metadata": example.get("metadata", {})
        })
    
    print(f"  Loaded {len(prompts)} prompts from test set")
    
    # Load validation set if provided (for larger AI pool)
    if val_file:
        print(f"Loading validation examples from {val_file}...")
        val_data = load_jsonl(val_file)
        for example in val_data:
            prompts.append({
                "prompt": example["input_text"],
                "metadata": example.get("metadata", {})
            })
        print(f"  Loaded {len(val_data)} prompts from validation set")
        print(f"  Total prompts available: {len(prompts)}")
    
    # Sample pools
    random.seed(seed)
    total_needed = ai_pool_size + human_pool_size
    
    if len(prompts) < total_needed:
        print(f"Warning: Only {len(prompts)} prompts available, but {total_needed} requested")
        print(f"  Using all available prompts and sampling with replacement if needed")
        
        if len(prompts) < human_pool_size:
            # Ensure we at least have enough for human pool
            human_pool = random.choices(prompts, k=human_pool_size)
            ai_pool = random.choices(prompts, k=ai_pool_size)
        else:
            # Split what we have
            random.shuffle(prompts)
            human_pool = prompts[:human_pool_size]
            ai_pool = prompts[human_pool_size:human_pool_size + ai_pool_size]
            if len(ai_pool) < ai_pool_size:
                # Sample with replacement to reach target
                ai_pool.extend(random.choices(prompts, k=ai_pool_size - len(ai_pool)))
    else:
        # We have enough prompts
        random.shuffle(prompts)
        human_pool = prompts[:human_pool_size]
        ai_pool = prompts[human_pool_size:human_pool_size + ai_pool_size]
        # Save remaining prompts for final evaluation (completely held-out)
        unused_prompts = prompts[human_pool_size + ai_pool_size:]
    
    print(f"  Created AI pool: {len(ai_pool)} prompts")
    print(f"  Created Human pool: {len(human_pool)} prompts")
    
    # Report unused prompts for final evaluation
    if len(prompts) > total_needed:
        print(f"  Held-out for final eval: {len(unused_prompts)} prompts (never used for preferences)")
    else:
        unused_prompts = []
    
    return ai_pool, human_pool, unused_prompts


def generate_two_responses(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> List[str]:
    """
    Generate two different responses for the same prompt.
    
    Returns:
        List of two response strings
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    responses = []
    for seed_offset in range(2):
        # Use different seeds for different responses
        set_seed(42 + seed_offset)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part (not the prompt)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        response = full_text[len(prompt):].strip()
        responses.append(response)
    
    return responses

def generate_label_response(
        model,
        prompt,
        responses,
        temperature: float = 0.7,
        max_tokens: int = 5
):
    label_query = (
        "You are a strict preference judge.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Response 1:\n{responses[0]}\n\n"
        f"Response 2:\n{responses[1]}\n\n"
        "Which response is the better argument?\n"
        "Answer with ONLY one character: 1 or 2."
    )

    messages = {"role": "user", "content": label_query}

    response = client.message.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = response.content[0].text.strip()
    if answer != "1" and answer != "2":
        raise ValueError(f"Unexpected response: {answer}")

    return int(answer)


def generate_sft_preference_data(
    model,
    tokenizer,
    prompts: List[Dict],
    output_file: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    device: str = "cuda"
):
    """
    Generate 2 responses per prompt and save to file.
    """
    results = []
    
    print(f"Generating responses for {len(prompts)} prompts...")
    for prompt_data in tqdm(prompts):
        prompt = prompt_data["prompt"]
        
        try:
            responses = generate_two_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                device=device
            )
            
            results.append({
                "prompt": prompt,
                "response_1": responses[0],
                "response_2": responses[1],
                "metadata": prompt_data.get("metadata", {})
            })
        except Exception as e:
            print(f"\nError generating for prompt: {e}")
            continue
    
    print(f"Generated {len(results)} preference pairs")
    
    # Save results
    save_jsonl(results, output_file)
    print(f"Saved to {output_file}")


def generate_dpo_preference_data(
        model,
        label_model,
        tokenizer,
        prompts: List[Dict],
        output_file: str,
        model_temperature: float = 0.7,
        label_temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        max_label_tokens: int = 5,
        device: str = "cuda"
):
    """
    For each prompt, generate 2 responses and label them.
    """
    results = []

    print(f"Generating labels for {len(prompts)} prompts...")
    for prompt_data in tqdm(prompts):
        prompt = prompt_data["prompt"]

        try:
            responses = generate_two_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=model_temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                device=device
            )

            answer = generate_label_response(
                model=label_model,
                prompt=prompt,
                responses=responses,
                temperature=label_temperature,
                max_tokens=max_label_tokens
            )

            if answer == 1:
                results.append({
                    "prompt": prompt,
                    "chosen": responses[0],
                    "rejected": responses[1],
                    "metadata": prompt_data.get("metadata", {})
                })

            else:
                results.append({
                    "prompt": prompt,
                    "chosen": responses[1],
                    "rejected": responses[0],
                    "metadata": prompt_data.get("metadata", {})
                })

        except Exception as e:
            print(f"\nError generating for prompt: {e}")
            continue

    print(f"Generated {len(results)} preference pairs")

    # Save results
    save_jsonl(results, output_file)
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate preference data from SFT model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/qwen-sft/final",
        help="Path to trained SFT model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/sft_test.jsonl",
        help="Test file to extract prompts from"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/sft_val.jsonl",
        help="Validation file to extract additional prompts from (optional)"
    )
    parser.add_argument(
        "--use-val-set",
        action="store_true",
        default=True,
        help="Use validation set for preference generation (default: True)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/preferences",
        help="Output directory for preference data"
    )
    parser.add_argument(
        "--ai-pool-size",
        type=int,
        default=2150,
        help="Number of prompts for RLAIF pool (default: 2150)"
    )
    parser.add_argument(
        "--human-pool-size",
        type=int,
        default=200,
        help="Number of prompts for RLHF pool (default: 200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Extract prompt pools from test (and val) set
    val_file = args.val_file if args.use_val_set else None
    ai_pool, human_pool, unused_pool = extract_prompts_from_test_set(
        test_file=args.test_file,
        val_file=val_file,
        ai_pool_size=args.ai_pool_size,
        human_pool_size=args.human_pool_size,
        seed=args.seed
    )
    
    # Save prompt pools
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ai_prompts_file = output_dir / "ai_pool_prompts.jsonl"
    human_prompts_file = output_dir / "human_pool_prompts.jsonl"
    final_eval_file = output_dir / "final_eval_prompts.jsonl"
    
    save_jsonl(ai_pool, str(ai_prompts_file))
    save_jsonl(human_pool, str(human_prompts_file))
    
    print(f"\nSaved prompt pools:")
    print(f"  AI pool: {ai_prompts_file}")
    print(f"  Human pool: {human_prompts_file}")
    
    # Save held-out prompts for final evaluation (if any)
    if unused_pool:
        save_jsonl(unused_pool, str(final_eval_file))
        print(f"  Final eval (from val+test): {final_eval_file} ({len(unused_pool)} prompts)")
        print(f"\n  ⭐ These {len(unused_pool)} prompts are NEVER used for preference generation.")
        print(f"     Use them for unbiased final evaluation of all models!")
    else:
        print(f"\n  Note: No unused prompts from val+test (all {len(ai_pool) + len(human_pool)} used for preferences)")
        print(f"  💡 The 115 final eval examples were reserved separately in Step 4 (final_eval_reserved.jsonl)")
    
    # Load model
    print(f"\nLoading trained SFT model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_trained_model(
        base_model_name=args.base_model,
        adapter_path=args.model_path,
        device_map=device
    )
    model.eval()
    
    # Generate responses for AI pool
    print("\n" + "="*80)
    print("Generating responses for AI pool...")
    print("="*80)
    ai_responses_file = output_dir / "ai_pool_responses.jsonl"
    generate_sft_preference_data(
        model=model,
        tokenizer=tokenizer,
        prompts=ai_pool,
        output_file=str(ai_responses_file),
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    # Generate responses for human pool
    print("\n" + "="*80)
    print("Generating responses for human pool...")
    print("="*80)
    human_responses_file = output_dir / "human_pool_responses.jsonl"
    generate_sft_preference_data(
        model=model,
        tokenizer=tokenizer,
        prompts=human_pool,
        output_file=str(human_responses_file),
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    print("\n" + "="*80)
    print("Preference data generation complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  AI pool prompts: {ai_prompts_file}")
    print(f"  AI pool responses: {ai_responses_file}")
    print(f"  Human pool prompts: {human_prompts_file}")
    print(f"  Human pool responses: {human_responses_file}")


if __name__ == "__main__":
    main()

