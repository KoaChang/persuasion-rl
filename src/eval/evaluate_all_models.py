#!/usr/bin/env python3
"""
Comprehensive evaluation of all 4 models:
- Base (Qwen2.5-0.5B-Instruct)
- SFT
- RLHF-only
- RLAIF→RLHF

Metrics:
1. Claude grader ranking scores (4-way ranking)
2. OpenAI embedding similarity to oracle (Claude 4.5 Sonnet)

Usage:
    python src/eval/evaluate_all_models.py [--config configs/eval_config.yaml]
"""

import argparse
import yaml
import json
import os
import sys
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_utils import load_base_model, load_trained_model
from src.eval.grader import create_claude_grader, create_openai_embedder


def load_all_models(config, tokenizer):
    """Load all 4 models."""
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    models = {}

    # Base model
    print("\n[1/4] Loading base model...")
    models['base'] = load_base_model(
        config['base_model'],
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✓ Base model loaded")

    # SFT model
    print("\n[2/4] Loading SFT model...")
    models['sft'] = load_trained_model(
        config['base_model'],
        config['sft_model_path'],
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✓ SFT model loaded")

    # RLHF-only model
    print("\n[3/4] Loading RLHF-only model...")
    if os.path.exists(config['rlhf_model_path']):
        models['rlhf'] = load_trained_model(
            config['base_model'],
            config['rlhf_model_path'],
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ RLHF model loaded")
    else:
        print("⚠️  RLHF model not found, skipping...")
        models['rlhf'] = None

    # RLAIF→RLHF model
    print("\n[4/4] Loading RLAIF→RLHF model...")
    if os.path.exists(config['rlaif_rlhf_model_path']):
        models['rlaif_rlhf'] = load_trained_model(
            config['base_model'],
            config['rlaif_rlhf_model_path'],
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ RLAIF→RLHF model loaded")
    else:
        print("⚠️  RLAIF→RLHF model not found, skipping...")
        models['rlaif_rlhf'] = None

    # Filter out None models
    models = {k: v for k, v in models.items() if v is not None}

    print(f"\n✓ Loaded {len(models)} models: {', '.join(models.keys())}")

    return models


def generate_response(model, tokenizer, prompt, gen_config):
    """Generate a single response from a model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            do_sample=gen_config['do_sample'],
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated portion (after prompt)
    response = response[len(prompt):].strip()

    return response


def generate_oracle_response(grader, prompt, temperature=0.7):
    """Generate oracle response using Claude 4.5 Sonnet."""
    # Extract context from the prompt (remove system message and task instruction)
    # The prompt format is: [SYSTEM]...[CONTEXT]...[TASK]...[RESPONSE]

    oracle_prompt = f"""You are a respectful assistant trying to persuade the other person using honest, well-reasoned arguments.

{prompt}

Write a persuasive response that:
- Directly addresses the other person's concerns or position
- Uses clear, coherent reasoning
- Maintains a respectful, non-manipulative tone
- Provides evidence or logical arguments where appropriate
- Avoids manipulation, false claims, or emotional exploitation

Response:"""

    # Enforce rate limiting
    if grader.rate_limiter:
        grader.rate_limiter.wait_if_needed()

    message = grader.client.messages.create(
        model=grader.model,
        max_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": oracle_prompt}]
    )

    return message.content[0].text.strip()


def save_progress(results, output_file):
    """Save results to file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of all models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_config.yaml',
        help='Evaluation config file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results file'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help='Maximum number of examples to evaluate (for testing)'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    eval_config = config['evaluation']
    gen_config = config['generation']
    oracle_config = config['oracle']
    batch_config = config['batch']

    # Create output directory
    os.makedirs(eval_config['output_dir'], exist_ok=True)
    output_file = os.path.join(eval_config['output_dir'], 'final_evaluation.json')

    # Load tokenizer
    print(f"\nLoading tokenizer from {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    models = load_all_models(config, tokenizer)
    model_names = list(models.keys())

    # Load test dataset
    print(f"\nLoading test dataset from {eval_config['test_file']}...")
    dataset = load_dataset("json", data_files=eval_config['test_file'])["train"]

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"Loaded {len(dataset)} test examples")

    # Load API clients
    print("\nInitializing API clients...")
    try:
        grader = create_claude_grader()
        embedder = create_openai_embedder()
        print("✓ API clients initialized")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your API keys in configs/api_config.yaml")
        return

    # Initialize results
    if args.resume and os.path.exists(output_file):
        print(f"\nResuming from {output_file}...")
        with open(output_file, 'r') as f:
            results = json.load(f)
        completed = len(results['per_example'])
        print(f"Found {completed} completed examples")
    else:
        results = {
            "config": config,
            "grader_scores": {name: 0 for name in model_names},
            "similarity_scores": {name: [] for name in model_names},
            "per_example": []
        }
        completed = 0

    # Set random seed for position randomization
    if eval_config['randomize_positions']:
        random.seed(eval_config['random_seed'])

    # Evaluate each test example
    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Examples: {len(dataset) - completed} remaining")
    print(f"Save progress every: {batch_config['save_every']} examples")
    print()

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating", initial=completed)):
        if idx < completed:
            continue  # Skip already completed

        prompt = example['input_text']

        # Generate responses from all models
        responses = {}
        print(f"\n[Example {idx + 1}/{len(dataset)}]")
        print("Generating responses...")

        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ", flush=True)
            responses[model_name] = generate_response(
                model, tokenizer, prompt, gen_config
            )
            print("✓")

        # Generate oracle response
        if oracle_config['generate']:
            print("  oracle (Claude 4.5)...", end=" ", flush=True)
            oracle = generate_oracle_response(
                grader, prompt, oracle_config['temperature']
            )
            print("✓")
        else:
            oracle = example.get('response', '')  # Use ground truth if oracle not generated

        # Randomly assign responses to positions A/B/C/D
        if eval_config['randomize_positions']:
            shuffled_names = model_names.copy()
            random.shuffle(shuffled_names)
        else:
            shuffled_names = model_names

        position_map = {chr(65 + i): shuffled_names[i] for i in range(len(shuffled_names))}

        labeled_responses = {
            pos: responses[model_name]
            for pos, model_name in position_map.items()
        }

        # Get grader ranking
        print("Getting grader ranking...", end=" ", flush=True)
        ranking, explanation = grader.rank_responses(prompt, labeled_responses)
        scores = grader.assign_scores(ranking)
        print("✓")

        # Map back to model names
        model_scores = {
            position_map[pos]: score
            for pos, score in scores.items()
        }

        # Accumulate grader scores
        for model_name, score in model_scores.items():
            results['grader_scores'][model_name] += score

        # Compute similarity scores
        print("Computing similarity scores...", end=" ", flush=True)
        similarities = {}
        for model_name, response in responses.items():
            sim = embedder.compute_similarity(response, oracle)
            similarities[model_name] = sim
            results['similarity_scores'][model_name].append(sim)
        print("✓")

        # Store per-example results
        results['per_example'].append({
            "example_id": idx,
            "prompt": prompt,
            "responses": responses,
            "oracle": oracle,
            "grader_ranking": ranking,
            "grader_scores": model_scores,
            "grader_explanation": explanation,
            "similarity_scores": similarities,
            "position_map": position_map
        })

        # Save progress periodically
        if (idx + 1) % batch_config['save_every'] == 0:
            print(f"Saving progress ({idx + 1}/{len(dataset)})...")
            save_progress(results, output_file)

    # Compute final metrics
    num_examples = len(results['per_example'])
    max_possible_score = 3 * num_examples

    results['summary'] = {
        "num_examples": num_examples,
        "models_evaluated": model_names,
        "grader_normalized_scores": {
            model: score / max_possible_score if max_possible_score > 0 else 0
            for model, score in results['grader_scores'].items()
        },
        "average_similarity_scores": {
            model: sum(scores) / len(scores) if scores else 0
            for model, scores in results['similarity_scores'].items()
        }
    }

    # Save final results
    save_progress(results, output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(f"\nGrader Ranking Scores (normalized to 0-1):")
    for model, score in sorted(
        results['summary']['grader_normalized_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {model:15s}: {score:.3f}")

    print(f"\nAverage Similarity to Oracle:")
    for model, score in sorted(
        results['summary']['average_similarity_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {model:15s}: {score:.3f}")

    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
