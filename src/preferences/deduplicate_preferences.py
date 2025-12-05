#!/usr/bin/env python3
"""
Remove duplicate prompts from preference datasets.
Keeps the first occurrence of each prompt.
"""

import json
from pathlib import Path


def deduplicate_preferences(input_file: str, output_file: str = None):
    """Remove duplicates from a JSONL preference file."""
    if output_file is None:
        output_file = input_file  # Overwrite in place
    
    preferences = []
    with open(input_file, 'r') as f:
        for line in f:
            preferences.append(json.loads(line))
    
    # Deduplicate by prompt, keeping first occurrence
    seen_prompts = set()
    deduplicated = []
    duplicates_removed = 0
    
    for item in preferences:
        prompt = item.get('prompt', '')
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            deduplicated.append(item)
        else:
            duplicates_removed += 1
    
    # Write deduplicated data
    with open(output_file, 'w') as f:
        for item in deduplicated:
            f.write(json.dumps(item) + '\n')
    
    return len(preferences), len(deduplicated), duplicates_removed


def main():
    data_dir = Path('data/preferences')
    
    # Deduplicate AI pool
    ai_file = data_dir / 'ai_pool_labeled.jsonl'
    if ai_file.exists():
        original, remaining, removed = deduplicate_preferences(str(ai_file))
        print(f"AI Pool: {original} -> {remaining} examples ({removed} duplicates removed)")
    
    # Deduplicate human pool
    human_file = data_dir / 'human_pool_labeled.jsonl'
    if human_file.exists():
        original, remaining, removed = deduplicate_preferences(str(human_file))
        print(f"Human Pool: {original} -> {remaining} examples ({removed} duplicates removed)")
    
    print("\n✓ Deduplication complete. Run validate_preferences.py to verify and recreate splits.")


if __name__ == '__main__':
    main()

