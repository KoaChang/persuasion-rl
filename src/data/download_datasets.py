"""
Download CMV (ChangeMyView) and PersuasionForGood datasets using ConvoKit.
"""

import os
import argparse
from pathlib import Path
from convokit import Corpus, download


def download_cmv(output_dir: Path):
    """Download the Winning Arguments (CMV) corpus from ConvoKit."""
    print("Downloading CMV (Winning Arguments) corpus...")
    try:
        cmv_corpus = Corpus(download("winning-args-corpus"))
        
        # Save the corpus to disk
        cmv_path = output_dir / "cmv"
        cmv_path.mkdir(parents=True, exist_ok=True)
        cmv_corpus.dump("winning-args-corpus", str(cmv_path))
        
        print(f"✓ CMV corpus downloaded to {cmv_path}")
        print(f"  Total conversations: {len(list(cmv_corpus.iter_conversations()))}")
        print(f"  Total utterances: {len(list(cmv_corpus.iter_utterances()))}")
        
        return cmv_corpus
    except Exception as e:
        print(f"✗ Error downloading CMV corpus: {e}")
        raise


def download_persuasionforgood(output_dir: Path):
    """Download the PersuasionForGood corpus from ConvoKit."""
    print("\nDownloading PersuasionForGood corpus...")
    try:
        p4g_corpus = Corpus(download("persuasionforgood-corpus"))
        
        # Save the corpus to disk
        p4g_path = output_dir / "persuasionforgood"
        p4g_path.mkdir(parents=True, exist_ok=True)
        p4g_corpus.dump("persuasionforgood-corpus", str(p4g_path))
        
        print(f"✓ PersuasionForGood corpus downloaded to {p4g_path}")
        print(f"  Total conversations: {len(list(p4g_corpus.iter_conversations()))}")
        print(f"  Total utterances: {len(list(p4g_corpus.iter_utterances()))}")
        
        return p4g_corpus
    except Exception as e:
        print(f"✗ Error downloading PersuasionForGood corpus: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download persuasion datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cmv", "persuasionforgood", "all"],
        default="all",
        help="Which dataset(s) to download"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}\n")
    
    if args.dataset in ["cmv", "all"]:
        download_cmv(output_dir)
    
    if args.dataset in ["persuasionforgood", "all"]:
        download_persuasionforgood(output_dir)
    
    print("\n✓ Dataset download complete!")


if __name__ == "__main__":
    main()

