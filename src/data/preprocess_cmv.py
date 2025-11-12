"""
Preprocess the CMV (ChangeMyView) dataset.
Extract delta-winning comments and their context.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from convokit import Corpus
from transformers import AutoTokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_delta_winning_examples(
    corpus: Corpus,
    tokenizer,
    min_tokens: int = 20,
    max_tokens: int = 1024,
    max_context_tokens: int = 512
) -> List[Dict]:
    """
    Extract examples where a comment won a delta.
    
    Returns list of dicts with:
    - context: conversation history (post + parent comments)
    - response: the delta-winning comment
    - metadata: additional info
    """
    examples = []
    
    print("Processing CMV conversations...")
    for conversation in corpus.iter_conversations():
        # Get all utterances in chronological order
        # Handle cases where timestamps might be None
        try:
            utterances = list(conversation.get_chronological_utterance_list())
        except ValueError:
            # If timestamp sorting fails, just get all utterances
            utterances = list(conversation.iter_utterances())
        
        # Find delta-winning comments
        for utt in utterances:
            # Check if this utterance won a delta
            # success field is 1 for delta-winning comments, None or 0 otherwise
            if utt.meta.get('success') == 1:
                response_text = utt.text.strip()
                
                # Filter by response length
                response_tokens = count_tokens(response_text, tokenizer)
                if response_tokens < min_tokens or response_tokens > max_tokens:
                    continue
                
                # Build context: original post + reply chain leading to this comment
                context_parts = []
                
                # Add original post (root) - find utterance with no reply_to
                root = None
                for u in utterances:
                    if u.reply_to is None:
                        root = u
                        break
                
                if root and root.text:
                    context_parts.append(f"Original Post: {root.text.strip()}")
                
                # Add parent comments in the reply chain
                current = utt
                parent_chain = []
                while current.reply_to is not None:
                    parent_id = current.reply_to
                    try:
                        parent = corpus.get_utterance(parent_id)
                        if parent.id != root.id and parent.text:
                            parent_chain.insert(0, parent.text.strip())
                        current = parent
                    except:
                        break
                
                # Add parent chain to context
                for i, parent_text in enumerate(parent_chain):
                    context_parts.append(f"Comment {i+1}: {parent_text}")
                
                context = "\n\n".join(context_parts)
                
                # Truncate context if too long
                context_tokens = count_tokens(context, tokenizer)
                if context_tokens > max_context_tokens:
                    # Simple truncation by taking first N characters
                    # More sophisticated truncation could be implemented
                    char_ratio = max_context_tokens / context_tokens
                    context = context[:int(len(context) * char_ratio * 0.9)]
                    context += "\n[...context truncated...]"
                
                examples.append({
                    "context": context,
                    "response": response_text,
                    "metadata": {
                        "source": "cmv",
                        "conversation_id": conversation.id,
                        "utterance_id": utt.id,
                        "response_tokens": response_tokens,
                        "context_tokens": count_tokens(context, tokenizer)
                    }
                })
    
    print(f"Extracted {len(examples)} delta-winning examples from CMV")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Preprocess CMV dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/cmv",
        help="Directory with downloaded CMV corpus"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/cmv_examples.jsonl",
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
    
    print(f"Loading CMV corpus from {input_dir}...")
    corpus = Corpus(filename=str(input_dir / "winning-args-corpus"))
    
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Extract examples
    examples = extract_delta_winning_examples(
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

