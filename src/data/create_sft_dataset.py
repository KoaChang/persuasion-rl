"""
Combine CMV and PersuasionForGood datasets into a unified SFT training dataset.
Apply prompt template and create train/val/test splits.
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


PROMPT_TEMPLATE = """[SYSTEM] You are a respectful assistant trying to persuade the other person using honest, well-reasoned arguments. You should acknowledge uncertainty and avoid manipulation.

[CONTEXT]
{context}

[TASK]
Write the next message that continues the conversation and aims to persuade the other person.

[RESPONSE]
{response}"""


PROMPT_TEMPLATE_NO_RESPONSE = """[SYSTEM] You are a respectful assistant trying to persuade the other person using honest, well-reasoned arguments. You should acknowledge uncertainty and avoid manipulation.

[CONTEXT]
{context}

[TASK]
Write the next message that continues the conversation and aims to persuade the other person.

[RESPONSE]
"""


def load_examples(file_path: Path) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def create_sft_example(example: Dict) -> Dict:
    """
    Convert a raw example into SFT format with prompt template.
    
    Returns dict with:
    - input_text: prompt without response (for generation)
    - full_text: prompt with response (for training)
    - metadata: preserved from original
    """
    context = example['context']
    response = example['response']
    
    # Full text for training (input + target)
    full_text = PROMPT_TEMPLATE.format(context=context, response=response)
    
    # Input only (for generation/inference)
    input_text = PROMPT_TEMPLATE_NO_RESPONSE.format(context=context)
    
    return {
        "input_text": input_text,
        "full_text": full_text,
        "response": response,
        "metadata": example['metadata']
    }


def split_dataset(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Split dataset into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]
    
    return train, val, test


def save_jsonl(examples: List[Dict], output_path: Path):
    """Save examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Create SFT dataset from preprocessed examples")
    parser.add_argument(
        "--cmv-file",
        type=str,
        default="data/processed/cmv_examples.jsonl",
        help="Path to preprocessed CMV examples"
    )
    parser.add_argument(
        "--p4g-file",
        type=str,
        default="data/processed/p4g_examples.jsonl",
        help="Path to preprocessed PersuasionForGood examples"
    )
    parser.add_argument(
        "--use-cmv",
        action="store_true",
        default=True,
        help="Use CMV dataset (default: True)"
    )
    parser.add_argument(
        "--no-cmv",
        action="store_false",
        dest="use_cmv",
        help="Skip CMV dataset"
    )
    parser.add_argument(
        "--use-p4g",
        action="store_true",
        default=False,
        help="Use PersuasionForGood dataset (default: False)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train/val/test splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=11750,
        help="Maximum number of examples to use (default: 11750, reserves 115 for final eval)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    all_examples = []
    
    # Load CMV examples if enabled
    if args.use_cmv:
        print("Loading CMV examples...")
        try:
            cmv_examples = load_examples(Path(args.cmv_file))
            print(f"  ✓ Loaded {len(cmv_examples)} CMV examples")
            all_examples.extend(cmv_examples)
        except FileNotFoundError:
            print(f"  ✗ CMV file not found: {args.cmv_file}")
            print("  Run: python src/data/preprocess_cmv.py first")
    else:
        print("Skipping CMV dataset (--no-cmv flag set)")
    
    # Load P4G examples if enabled
    if args.use_p4g:
        print("Loading PersuasionForGood examples...")
        try:
            p4g_examples = load_examples(Path(args.p4g_file))
            print(f"  ✓ Loaded {len(p4g_examples)} P4G examples")
            all_examples.extend(p4g_examples)
        except FileNotFoundError:
            print(f"  ✗ P4G file not found: {args.p4g_file}")
            print("  Run: python src/data/preprocess_persuasionforgood.py first")
    else:
        print("Skipping PersuasionForGood dataset (use --use-p4g to include)")
    
    if not all_examples:
        print("\n✗ Error: No examples loaded! Enable at least one dataset.")
        return
    
    print(f"\nTotal examples loaded: {len(all_examples)}")
    
    # Apply max examples limit if specified
    if args.max_examples and args.max_examples > 0 and args.max_examples < len(all_examples):
        random.seed(args.seed)
        all_examples = random.sample(all_examples, args.max_examples)
        print(f"Sampled {len(all_examples)} examples (limited by --max-examples={args.max_examples})")
    elif args.max_examples and args.max_examples > 0:
        print(f"Using all {len(all_examples)} examples (--max-examples={args.max_examples} >= available data)")
    
    # Convert to SFT format
    print("\nApplying prompt template...")
    sft_examples = [create_sft_example(ex) for ex in all_examples]
    
    # Split into train/val/test
    print("Creating train/val/test splits...")
    train, val, test = split_dataset(
        sft_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(train)} examples ({args.train_ratio*100:.1f}%)")
    print(f"  Val: {len(val)} examples ({args.val_ratio*100:.1f}%)")
    print(f"  Test: {len(test)} examples ({args.test_ratio*100:.1f}%)")
    
    # Save splits
    print("\nSaving splits...")
    save_jsonl(train, output_dir / "sft_train.jsonl")
    save_jsonl(val, output_dir / "sft_val.jsonl")
    save_jsonl(test, output_dir / "sft_test.jsonl")
    
    print(f"✓ Saved train/val/test splits to {output_dir}/")
    
    # Print example
    print("\n" + "="*80)
    print("Example SFT training instance:")
    print("="*80)
    if train:
        example = train[0]
        print(example['full_text'][:1000])
        if len(example['full_text']) > 1000:
            print(f"\n[...truncated, total length: {len(example['full_text'])} chars]")
    print("="*80)


if __name__ == "__main__":
    main()

