"""
export_vocab.py

Extracts vocab.txt from the same tokenizer used in the training script
(distilbert-base-uncased, or your fine-tuned checkpoint).

Two modes:
1. From the base pretrained tokenizer (before training) -> vocab.txt
2. From your saved fine-tuned model dir (./platform_classifier_model) -> vocab.txt
   (useful if you want the vocab that matches the exact tokenizer files
   you saved after training)

Usage:
    python export_vocab.py --source base
    python export_vocab.py --source finetuned --model_dir ./platform_classifier_model
"""

import argparse
from pathlib import Path
from transformers import DistilBertTokenizerFast


def export_vocab(tokenizer_path: str, output_dir: str = "./vocab_output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

    # save_vocabulary writes vocab.txt (and returns the path(s) written)
    saved_files = tokenizer.save_vocabulary(str(output_dir))

    print(f"✅ vocab.txt exported to: {output_dir / 'vocab.txt'}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["base", "finetuned"],
        default="base",
        help="Use the base pretrained tokenizer, or your fine-tuned model dir",
    )
    parser.add_argument(
        "--model_dir",
        default="./platform_classifier_model",
        help="Path to your saved fine-tuned model/tokenizer dir (used when --source finetuned)",
    )
    parser.add_argument(
        "--output_dir",
        default="./vocab_output",
        help="Where to write vocab.txt",
    )
    args = parser.parse_args()

    tokenizer_path = "distilbert-base-uncased" if args.source == "base" else args.model_dir
    export_vocab(tokenizer_path, args.output_dir)