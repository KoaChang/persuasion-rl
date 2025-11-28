#!/usr/bin/env python3
"""
Interactive CLI for human labeling of preference pairs.

This script:
1. Loads data/preferences/human_pool_responses.jsonl (200 preference pairs)
2. Displays prompt and 2 responses in a clear format
3. User selects which response is more persuasive
4. Saves labeled preferences to data/preferences/human_pool_labeled.jsonl

Usage:
    python src/preferences/label_human_preferences.py [--resume]
"""

import argparse
import json
import os
import sys
from pathlib import Path


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


def display_preference_pair(idx: int, total: int, item: dict):
    """Display a preference pair for human labeling."""
    # Clear screen (works on Unix and Windows)
    os.system('clear' if os.name == 'posix' else 'cls')

    print("=" * 80)
    print(f"HUMAN PREFERENCE LABELING")
    print("=" * 80)
    print(f"Progress: {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)")
    print("=" * 80)
    print()

    # Extract prompt (remove the [RESPONSE] marker at the end)
    prompt = item['prompt']
    if '[RESPONSE]\n' in prompt:
        prompt = prompt.split('[RESPONSE]\n')[0] + '[RESPONSE]'

    print("CONTEXT:")
    print("-" * 80)
    print(prompt)
    print()

    print("=" * 80)
    print("RESPONSE 1:")
    print("-" * 80)
    print(item['response_1'])
    print()

    print("=" * 80)
    print("RESPONSE 2:")
    print("-" * 80)
    print(item['response_2'])
    print()

    print("=" * 80)


def get_user_choice():
    """Get user's preference choice."""
    while True:
        print("\nWhich response is MORE PERSUASIVE?")
        print("  [1] Response 1")
        print("  [2] Response 2")
        print("  [E] Equal (will randomly assign)")
        print("  [S] Skip this example")
        print("  [Q] Quit and save progress")
        print()

        choice = input("Your choice: ").strip().upper()

        if choice in ['1', '2', 'E', 'S', 'Q']:
            return choice
        else:
            print("Invalid choice. Please enter 1, 2, E, S, or Q.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for human preference labeling"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/preferences/human_pool_responses.jsonl',
        help='Input file with response pairs'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/preferences/human_pool_labeled.jsonl',
        help='Output file for labeled preferences'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
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
        # Confirm overwrite
        confirm = input("Output file exists. Overwrite? [y/N]: ").strip().lower()
        if confirm == 'y':
            os.remove(args.output_file)
            print("Starting fresh (removed existing output file)")
        else:
            print("Exiting. Use --resume to continue from existing progress.")
            return

    # Skip already completed examples
    responses_to_label = responses[completed:]

    if len(responses_to_label) == 0:
        print("All examples already labeled!")
        return

    # Statistics
    stats = {
        'total': len(responses_to_label),
        'completed': 0,
        'response_1_preferred': 0,
        'response_2_preferred': 0,
        'equal': 0,
        'skipped': 0
    }

    print(f"\nStarting labeling of {len(responses_to_label)} preference pairs...")
    print("Press Enter to continue...")
    input()

    # Label preferences
    for idx, item in enumerate(responses_to_label):
        # Display preference pair
        display_preference_pair(completed + idx, total, item)

        # Get user choice
        choice = get_user_choice()

        if choice == 'Q':
            print("\nQuitting and saving progress...")
            break

        if choice == 'S':
            print("Skipping this example...")
            stats['skipped'] += 1
            input("Press Enter to continue...")
            continue

        # Determine chosen and rejected
        import random
        if choice == '1':
            chosen = item['response_1']
            rejected = item['response_2']
            preference = '1'
            stats['response_1_preferred'] += 1
        elif choice == '2':
            chosen = item['response_2']
            rejected = item['response_1']
            preference = '2'
            stats['response_2_preferred'] += 1
        elif choice == 'E':
            # Randomly assign
            if random.random() < 0.5:
                chosen = item['response_1']
                rejected = item['response_2']
                preference = '1'
                stats['response_1_preferred'] += 1
            else:
                chosen = item['response_2']
                rejected = item['response_1']
                preference = '2'
                stats['response_2_preferred'] += 1
            stats['equal'] += 1

        # Create labeled preference
        labeled_data = {
            'prompt': item['prompt'],
            'chosen': chosen,
            'rejected': rejected,
            'metadata': {
                **item.get('metadata', {}),
                'grader': 'human',
                'preference': preference,
                'is_equal': (choice == 'E')
            }
        }

        # Save to file
        save_labeled_preference(args.output_file, labeled_data)
        stats['completed'] += 1

        print(f"\nSaved! ({stats['completed']}/{stats['total']})")
        print("Press Enter to continue to next example...")
        input()

    # Final statistics
    print("\n" + "=" * 80)
    print("LABELING SESSION COMPLETE")
    print("=" * 80)
    print(f"Total labeled: {stats['completed']}/{stats['total']}")
    print(f"Response 1 preferred: {stats['response_1_preferred']} "
          f"({stats['response_1_preferred']/stats['completed']*100:.1f}%)" if stats['completed'] > 0 else "Response 1 preferred: 0")
    print(f"Response 2 preferred: {stats['response_2_preferred']} "
          f"({stats['response_2_preferred']/stats['completed']*100:.1f}%)" if stats['completed'] > 0 else "Response 2 preferred: 0")
    print(f"Equal (randomly assigned): {stats['equal']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"\nLabeled preferences saved to: {args.output_file}")

    remaining = stats['total'] - stats['completed']
    if remaining > 0:
        print(f"\n{remaining} examples remaining. Run with --resume to continue.")


if __name__ == '__main__':
    main()
