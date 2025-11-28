#!/usr/bin/env python3
"""
Label AI preference pool using Claude 4.5 Sonnet as grader.

This script:
1. Loads data/preferences/ai_pool_responses.jsonl (2,150 preference pairs)
2. For each pair, asks Claude which response is more persuasive
3. Saves labeled preferences to data/preferences/ai_pool_labeled.jsonl

Usage:
    python src/preferences/label_ai_preferences.py [--resume]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.api_clients import create_claude_grader


def load_responses(input_file: str):
    """Load preference pairs from JSONL file."""
    responses = []
    with open(input_file, 'r') as f:
        for line in f:
            responses.append(json.loads(line))
    return responses


def save_labeled_preference(output_file: str, labeled_data: dict):
    """Append a labeled preference to the output file."""
    with open(output_file, 'a') as f:
        f.write(json.dumps(labeled_data) + '\n')


def get_completed_count(output_file: str) -> int:
    """Count how many preferences have been labeled."""
    if not os.path.exists(output_file):
        return 0

    count = 0
    with open(output_file, 'r') as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Label AI preference pool using Claude grader"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/preferences/ai_pool_responses.jsonl',
        help='Input file with response pairs'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/preferences/ai_pool_labeled.jsonl',
        help='Output file for labeled preferences'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=100,
        help='Print progress every N examples'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help='Maximum number of examples to label (for testing)'
    )
    args = parser.parse_args()

    # Load responses
    print(f"Loading responses from {args.input_file}...")
    responses = load_responses(args.input_file)
    total = len(responses)
    print(f"Loaded {total} preference pairs")

    # Check for existing progress
    completed = 0
    if args.resume and os.path.exists(args.output_file):
        completed = get_completed_count(args.output_file)
        print(f"Resuming from {completed} completed examples")
    elif not args.resume and os.path.exists(args.output_file):
        # Remove existing file if not resuming
        os.remove(args.output_file)
        print("Starting fresh (removed existing output file)")

    # Limit examples if specified (for testing)
    if args.max_examples:
        responses = responses[:args.max_examples]
        total = len(responses)
        print(f"Limiting to {total} examples for testing")

    # Skip already completed examples
    responses_to_label = responses[completed:]

    if len(responses_to_label) == 0:
        print("All examples already labeled!")
        return

    # Initialize Claude grader
    print("Initializing Claude grader...")
    try:
        grader = create_claude_grader()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your Claude API key in configs/api_config.yaml")
        return

    # Statistics
    stats = {
        'total': len(responses_to_label),
        'completed': 0,
        'response_1_preferred': 0,
        'response_2_preferred': 0,
        'equal': 0,
        'errors': 0,
        'total_api_cost_estimate': 0.0
    }

    print(f"\nStarting labeling of {len(responses_to_label)} preference pairs...")
    print("=" * 60)

    # Label preferences
    start_time = time.time()

    for idx, item in enumerate(tqdm(responses_to_label, desc="Labeling preferences")):
        try:
            prompt = item['prompt']
            response_1 = item['response_1']
            response_2 = item['response_2']
            metadata = item.get('metadata', {})

            # Grade preference pair
            chosen, rejected, reasoning = grader.grade_preference_pair(
                prompt, response_1, response_2
            )

            # Determine which was preferred
            if chosen == response_1:
                preference = '1'
                stats['response_1_preferred'] += 1
            elif chosen == response_2:
                preference = '2'
                stats['response_2_preferred'] += 1
            else:
                # This shouldn't happen, but handle it
                preference = 'unknown'

            # Check if it was an EQUAL case (randomly assigned)
            is_equal = '[EQUAL - randomly assigned]' in reasoning
            if is_equal:
                stats['equal'] += 1

            # Create labeled preference
            labeled_data = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'metadata': {
                    **metadata,
                    'grader': 'claude-sonnet-4-5',
                    'preference': preference,
                    'is_equal': is_equal,
                    'reasoning': reasoning
                }
            }

            # Save to file
            save_labeled_preference(args.output_file, labeled_data)
            stats['completed'] += 1

            # Estimate API cost (rough estimate: ~500 tokens per request)
            # Claude Sonnet 4.5: $3/M input tokens, $15/M output tokens
            # Assume 400 input + 100 output = ~$1.65/1000 requests
            stats['total_api_cost_estimate'] += 0.00165

            # Print progress
            if (idx + 1) % args.save_every == 0:
                elapsed = time.time() - start_time
                rate = stats['completed'] / elapsed if elapsed > 0 else 0
                remaining = (stats['total'] - stats['completed']) / rate if rate > 0 else 0

                print(f"\nProgress: {stats['completed']}/{stats['total']}")
                print(f"  Response 1 preferred: {stats['response_1_preferred']} "
                      f"({stats['response_1_preferred']/stats['completed']*100:.1f}%)")
                print(f"  Response 2 preferred: {stats['response_2_preferred']} "
                      f"({stats['response_2_preferred']/stats['completed']*100:.1f}%)")
                print(f"  Equal (randomly assigned): {stats['equal']} "
                      f"({stats['equal']/stats['completed']*100:.1f}%)")
                print(f"  Rate: {rate:.1f} examples/sec")
                print(f"  Estimated time remaining: {remaining/60:.1f} minutes")
                print(f"  Estimated API cost so far: ${stats['total_api_cost_estimate']:.2f}")

        except Exception as e:
            print(f"\nError on example {completed + idx + 1}: {e}")
            stats['errors'] += 1

            # Save error info
            error_data = {
                'prompt': item['prompt'],
                'response_1': item['response_1'],
                'response_2': item['response_2'],
                'error': str(e)
            }
            with open(args.output_file + '.errors', 'a') as f:
                f.write(json.dumps(error_data) + '\n')

            # Continue to next example
            continue

    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)
    print(f"Total labeled: {stats['completed']}/{stats['total']}")
    print(f"Response 1 preferred: {stats['response_1_preferred']} "
          f"({stats['response_1_preferred']/stats['completed']*100:.1f}%)")
    print(f"Response 2 preferred: {stats['response_2_preferred']} "
          f"({stats['response_2_preferred']/stats['completed']*100:.1f}%)")
    print(f"Equal (randomly assigned): {stats['equal']} "
          f"({stats['equal']/stats['completed']*100:.1f}%)")
    print(f"Errors: {stats['errors']}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Average rate: {stats['completed']/elapsed:.2f} examples/sec")
    print(f"Estimated total API cost: ${stats['total_api_cost_estimate']:.2f}")
    print(f"\nLabeled preferences saved to: {args.output_file}")

    if stats['errors'] > 0:
        print(f"Errors saved to: {args.output_file}.errors")


if __name__ == '__main__':
    main()
