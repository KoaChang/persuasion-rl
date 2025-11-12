"""
Preprocess the PersuasionForGood dataset.
Extract persuader turns from dialogues.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from convokit import Corpus
from transformers import AutoTokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_persuader_examples(
    corpus: Corpus,
    tokenizer,
    min_tokens: int = 20,
    max_tokens: int = 1024,
    max_context_tokens: int = 512
) -> List[Dict]:
    """
    Extract examples where the persuader is making an argument.
    
    Returns list of dicts with:
    - context: conversation history up to the persuader's turn
    - response: the persuader's utterance
    - metadata: additional info
    """
    examples = []
    
    print("Processing PersuasionForGood conversations...")
    for conversation in corpus.iter_conversations():
        utterances = list(conversation.get_chronological_utterance_list())
        
        # Build context incrementally through the conversation
        for i, utt in enumerate(utterances):
            # Check if this is a persuader utterance
            # In PersuasionForGood, persuader is typically indicated in metadata
            speaker_role = utt.speaker.meta.get('role', '').lower()
            
            # Only extract persuader turns
            if 'persuader' not in speaker_role and 'er' not in speaker_role:
                continue
            
            response_text = utt.text.strip()
            
            # Filter by response length
            response_tokens = count_tokens(response_text, tokenizer)
            if response_tokens < min_tokens or response_tokens > max_tokens:
                continue
            
            # Build context from previous utterances
            context_parts = []
            for prev_utt in utterances[:i]:
                prev_role = prev_utt.speaker.meta.get('role', 'unknown')
                prev_text = prev_utt.text.strip()
                context_parts.append(f"{prev_role}: {prev_text}")
            
            if not context_parts:
                # Skip if no prior context (first utterance)
                continue
            
            context = "\n".join(context_parts)
            
            # Truncate context if too long
            context_tokens = count_tokens(context, tokenizer)
            if context_tokens > max_context_tokens:
                # Keep most recent context
                # Simple truncation from the beginning
                char_ratio = max_context_tokens / context_tokens
                truncate_point = int(len(context) * (1 - char_ratio))
                context = "[...earlier context truncated...]\n" + context[truncate_point:]
            
            examples.append({
                "context": context,
                "response": response_text,
                "metadata": {
                    "source": "persuasionforgood",
                    "conversation_id": conversation.id,
                    "utterance_id": utt.id,
                    "speaker_role": speaker_role,
                    "response_tokens": response_tokens,
                    "context_tokens": count_tokens(context, tokenizer)
                }
            })
    
    print(f"Extracted {len(examples)} persuader examples from PersuasionForGood")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Preprocess PersuasionForGood dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/persuasionforgood",
        help="Directory with downloaded PersuasionForGood corpus"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/p4g_examples.jsonl",
        help="Output file for processed examples"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=20,
        help="Minimum response length in tokens"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum response length in tokens"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=512,
        help="Maximum context length in tokens"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PersuasionForGood corpus from {input_dir}...")
    corpus = Corpus(filename=str(input_dir / "persuasionforgood-corpus"))
    
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Extract examples
    examples = extract_persuader_examples(
        corpus=corpus,
        tokenizer=tokenizer,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_context_tokens=args.max_context_tokens
    )
    
    # Save to JSONL
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(examples)} examples to {output_file}")
    
    # Print statistics
    if examples:
        avg_response_tokens = sum(e['metadata']['response_tokens'] for e in examples) / len(examples)
        avg_context_tokens = sum(e['metadata']['context_tokens'] for e in examples) / len(examples)
        print(f"\nStatistics:")
        print(f"  Average response tokens: {avg_response_tokens:.1f}")
        print(f"  Average context tokens: {avg_context_tokens:.1f}")


if __name__ == "__main__":
    main()

