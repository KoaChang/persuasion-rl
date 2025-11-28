#!/usr/bin/env python3
"""
Validate labeled preference datasets and create train/val splits for DPO.

This script:
1. Validates AI and human labeled preference files
2. Checks for required fields, duplicates, and biases
3. Creates train/val splits (90/10)
4. Generates validation report

Usage:
    python src/preferences/validate_preferences.py
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer


def load_preferences(file_path: str):
    """Load preferences from JSONL file."""
    preferences = []
    with open(file_path, 'r') as f:
        for line in f:
            preferences.append(json.loads(line))
    return preferences


def save_preferences(preferences, file_path: str):
    """Save preferences to JSONL file."""
    with open(file_path, 'w') as f:
        for item in preferences:
            f.write(json.dumps(item) + '\n')


def validate_preferences(preferences, name: str, tokenizer=None):
    """Validate preference dataset and return statistics."""
    stats = {
        'name': name,
        'total': len(preferences),
        'missing_fields': 0,
        'duplicates': 0,
        'preference_distribution': defaultdict(int),
        'equal_count': 0,
        'avg_chosen_length': 0,
        'avg_rejected_length': 0,
        'avg_chosen_tokens': 0,
        'avg_rejected_tokens': 0,
        'warnings': []
    }

    # Check for required fields
    required_fields = ['prompt', 'chosen', 'rejected', 'metadata']
    for idx, item in enumerate(preferences):
        for field in required_fields:
            if field not in item:
                stats['missing_fields'] += 1
                stats['warnings'].append(f"Example {idx}: Missing field '{field}'")

    # Check for duplicates (based on prompt)
    prompts_seen = set()
    for idx, item in enumerate(preferences):
        prompt = item.get('prompt', '')
        if prompt in prompts_seen:
            stats['duplicates'] += 1
            stats['warnings'].append(f"Example {idx}: Duplicate prompt")
        prompts_seen.add(prompt)

    # Compute preference distribution
    for item in preferences:
        preference = item.get('metadata', {}).get('preference', 'unknown')
        stats['preference_distribution'][preference] += 1

        is_equal = item.get('metadata', {}).get('is_equal', False)
        if is_equal:
            stats['equal_count'] += 1

    # Compute length statistics
    chosen_lengths = []
    rejected_lengths = []
    chosen_token_counts = []
    rejected_token_counts = []

    for item in preferences:
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')

        chosen_lengths.append(len(chosen))
        rejected_lengths.append(len(rejected))

        if tokenizer:
            chosen_tokens = len(tokenizer.encode(chosen))
            rejected_tokens = len(tokenizer.encode(rejected))
            chosen_token_counts.append(chosen_tokens)
            rejected_token_counts.append(rejected_tokens)

    stats['avg_chosen_length'] = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0
    stats['avg_rejected_length'] = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0

    if tokenizer and chosen_token_counts:
        stats['avg_chosen_tokens'] = sum(chosen_token_counts) / len(chosen_token_counts)
        stats['avg_rejected_tokens'] = sum(rejected_token_counts) / len(rejected_token_counts)

    # Check for length bias
    if stats['avg_chosen_length'] > stats['avg_rejected_length'] * 1.2:
        stats['warnings'].append(
            f"Potential length bias: chosen responses are {stats['avg_chosen_length'] / stats['avg_rejected_length']:.2f}x longer on average"
        )

    return stats


def create_splits(preferences, train_ratio=0.9, seed=42):
    """Create train/val splits."""
    random.seed(seed)
    shuffled = preferences.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    return train, val


def print_stats(stats):
    """Print validation statistics."""
    print(f"\n{'=' * 80}")
    print(f"Validation Report: {stats['name']}")
    print(f"{'=' * 80}")
    print(f"Total examples: {stats['total']}")
    print(f"Missing fields: {stats['missing_fields']}")
    print(f"Duplicates: {stats['duplicates']}")
    print()

    print("Preference Distribution:")
    for pref, count in sorted(stats['preference_distribution'].items()):
        percentage = count / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {pref}: {count} ({percentage:.1f}%)")
    print(f"  Equal (randomly assigned): {stats['equal_count']} ({stats['equal_count'] / stats['total'] * 100:.1f}%)")
    print()

    print("Length Statistics:")
    print(f"  Avg chosen length: {stats['avg_chosen_length']:.1f} characters")
    print(f"  Avg rejected length: {stats['avg_rejected_length']:.1f} characters")
    print(f"  Ratio (chosen/rejected): {stats['avg_chosen_length'] / stats['avg_rejected_length']:.2f}x")

    if stats['avg_chosen_tokens'] > 0:
        print(f"  Avg chosen tokens: {stats['avg_chosen_tokens']:.1f}")
        print(f"  Avg rejected tokens: {stats['avg_rejected_tokens']:.1f}")
        print(f"  Token ratio (chosen/rejected): {stats['avg_chosen_tokens'] / stats['avg_rejected_tokens']:.2f}x")

    if stats['warnings']:
        print()
        print(f"Warnings ({len(stats['warnings'])}):")
        for warning in stats['warnings'][:10]:  # Show first 10
            print(f"  - {warning}")
        if len(stats['warnings']) > 10:
            print(f"  ... and {len(stats['warnings']) - 10} more warnings")


def save_report(ai_stats, human_stats, output_file: str):
    """Save validation report to file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREFERENCE DATASET VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        for stats in [ai_stats, human_stats]:
            f.write(f"\n{'-' * 80}\n")
            f.write(f"{stats['name']}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"Total examples: {stats['total']}\n")
            f.write(f"Missing fields: {stats['missing_fields']}\n")
            f.write(f"Duplicates: {stats['duplicates']}\n\n")

            f.write("Preference Distribution:\n")
            for pref, count in sorted(stats['preference_distribution'].items()):
                percentage = count / stats['total'] * 100 if stats['total'] > 0 else 0
                f.write(f"  {pref}: {count} ({percentage:.1f}%)\n")
            f.write(f"  Equal: {stats['equal_count']} ({stats['equal_count'] / stats['total'] * 100:.1f}%)\n\n")

            f.write("Length Statistics:\n")
            f.write(f"  Avg chosen length: {stats['avg_chosen_length']:.1f} characters\n")
            f.write(f"  Avg rejected length: {stats['avg_rejected_length']:.1f} characters\n")
            f.write(f"  Ratio: {stats['avg_chosen_length'] / stats['avg_rejected_length']:.2f}x\n")

            if stats['avg_chosen_tokens'] > 0:
                f.write(f"  Avg chosen tokens: {stats['avg_chosen_tokens']:.1f}\n")
                f.write(f"  Avg rejected tokens: {stats['avg_rejected_tokens']:.1f}\n")
                f.write(f"  Token ratio: {stats['avg_chosen_tokens'] / stats['avg_rejected_tokens']:.2f}x\n")

            if stats['warnings']:
                f.write(f"\nWarnings ({len(stats['warnings'])}):\n")
                for warning in stats['warnings']:
                    f.write(f"  - {warning}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total RLAIF examples: {ai_stats['total']}\n")
        f.write(f"Total RLHF examples: {human_stats['total']}\n")
        f.write(f"RLAIF:RLHF ratio: {ai_stats['total'] / human_stats['total']:.2f}x\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate preference datasets and create train/val splits"
    )
    parser.add_argument(
        '--ai_file',
        type=str,
        default='data/preferences/ai_pool_labeled.jsonl',
        help='AI labeled preferences file'
    )
    parser.add_argument(
        '--human_file',
        type=str,
        default='data/preferences/human_pool_labeled.jsonl',
        help='Human labeled preferences file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/preferences',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Train split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling'
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer for token counting
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Load and validate AI preferences
    print(f"\nLoading AI preferences from {args.ai_file}...")
    ai_preferences = load_preferences(args.ai_file)
    ai_stats = validate_preferences(ai_preferences, "AI Pool (RLAIF)", tokenizer)
    print_stats(ai_stats)

    # Load and validate human preferences
    print(f"\nLoading human preferences from {args.human_file}...")
    human_preferences = load_preferences(args.human_file)
    human_stats = validate_preferences(human_preferences, "Human Pool (RLHF)", tokenizer)
    print_stats(human_stats)

    # Create splits for AI preferences
    print(f"\nCreating splits for AI preferences (train ratio: {args.train_ratio})...")
    ai_train, ai_val = create_splits(ai_preferences, args.train_ratio, args.seed)
    ai_train_file = os.path.join(args.output_dir, 'rlaif_train.jsonl')
    ai_val_file = os.path.join(args.output_dir, 'rlaif_val.jsonl')
    save_preferences(ai_train, ai_train_file)
    save_preferences(ai_val, ai_val_file)
    print(f"  Train: {len(ai_train)} examples -> {ai_train_file}")
    print(f"  Val: {len(ai_val)} examples -> {ai_val_file}")

    # Create splits for human preferences
    print(f"\nCreating splits for human preferences (train ratio: {args.train_ratio})...")
    human_train, human_val = create_splits(human_preferences, args.train_ratio, args.seed)
    human_train_file = os.path.join(args.output_dir, 'rlhf_train.jsonl')
    human_val_file = os.path.join(args.output_dir, 'rlhf_val.jsonl')
    save_preferences(human_train, human_train_file)
    save_preferences(human_val, human_val_file)
    print(f"  Train: {len(human_train)} examples -> {human_train_file}")
    print(f"  Val: {len(human_val)} examples -> {human_val_file}")

    # Save validation report
    report_file = os.path.join(args.output_dir, 'validation_report.txt')
    save_report(ai_stats, human_stats, report_file)
    print(f"\nValidation report saved to: {report_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Total RLAIF examples: {len(ai_preferences)} (train: {len(ai_train)}, val: {len(ai_val)})")
    print(f"Total RLHF examples: {len(human_preferences)} (train: {len(human_train)}, val: {len(human_val)})")
    print(f"RLAIF:RLHF ratio: {len(ai_preferences) / len(human_preferences):.2f}x")

    # Check for errors
    total_errors = ai_stats['missing_fields'] + ai_stats['duplicates'] + \
                   human_stats['missing_fields'] + human_stats['duplicates']

    if total_errors > 0:
        print(f"\n⚠️  WARNING: Found {total_errors} errors. Please review the validation report.")
    else:
        print("\n✓ No errors found. Datasets are ready for DPO training!")


if __name__ == '__main__':
    main()
