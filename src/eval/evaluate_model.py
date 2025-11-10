"""
Evaluate trained model by generating sample outputs.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.model_utils import load_trained_model


def load_jsonl(file_path: str, max_examples: int = None) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data.append(json.loads(line))
    return data


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """Generate a response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    response = full_text[len(prompt):].strip()
    return response


def evaluate_model_samples(
    model,
    tokenizer,
    test_examples: List[Dict],
    num_samples: int = 10,
    output_file: str = None,
    device: str = "cuda"
):
    """
    Generate responses for a sample of test examples and display them.
    """
    print(f"\nGenerating responses for {num_samples} test examples...\n")
    
    results = []
    
    for i, example in enumerate(test_examples[:num_samples]):
        print("="*80)
        print(f"Example {i+1}/{num_samples}")
        print("="*80)
        
        prompt = example["input_text"]
        ground_truth = example["response"]
        
        # Generate response
        generated = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device
        )
        
        print("\n[PROMPT]")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("\n[GROUND TRUTH RESPONSE]")
        print(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)
        print("\n[GENERATED RESPONSE]")
        print(generated[:500] + "..." if len(generated) > 500 else generated)
        print()
        
        results.append({
            "example_id": i,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": generated,
            "metadata": example.get("metadata", {})
        })
    
    # Save results if output file specified
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SFT model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/qwen-sft/final",
        help="Path to trained model"
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
        help="Test file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save results (optional)"
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
        help="Maximum new tokens"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    test_examples = load_jsonl(args.test_file)
    print(f"Loaded {len(test_examples)} test examples")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_trained_model(
        base_model_name=args.base_model,
        adapter_path=args.model_path,
        device_map=device
    )
    model.eval()
    
    print("Model loaded successfully!")
    
    # Generate and evaluate samples
    evaluate_model_samples(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        num_samples=args.num_samples,
        output_file=args.output_file,
        device=device
    )


if __name__ == "__main__":
    main()

