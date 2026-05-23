"""
Financial Foundation Model Pre-training — Decoder-Only (HuggingFace Trainer)

Port of the original NeMo AutoModel training script to run on Apple Silicon
(MPS) or any CPU/GPU using HuggingFace Transformers Trainer.

Original model architecture is preserved:
  - Llama decoder, ~29M params, hidden=512, 8 layers, GQA, SwiGLU, RMSNorm

Usage:
    Single device (Mac/CPU):
        python scripts/train_decoder_hf.py \
            --train_data data/decoder_corpus/train_corpus.txt \
            --val_data data/decoder_corpus/val_corpus.txt \
            --output_dir models/decoder-demo

    Override hyperparameters:
        python scripts/train_decoder_hf.py \
            --train_data data/decoder_corpus/train_corpus.txt \
            --max_steps 30 \
            --per_device_train_batch_size 16 \
            --learning_rate 2e-4
"""

import argparse
import sys
from pathlib import Path

BLUEPRINT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BLUEPRINT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train financial transaction foundation model (HF Trainer)"
    )
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training corpus (.txt)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation corpus (.txt)")
    parser.add_argument("--output_dir", type=str,
                        default="models/decoder-demo",
                        help="Output directory for checkpoints")
    parser.add_argument("--merchant_hash_size", type=int, default=2000)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.077)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=15)
    parser.add_argument("--eval_steps", type=int, default=15)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser.parse_args()


def main():
    import torch
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainingArguments,
    )

    args = parse_args()

    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    config = LlamaConfig(
        vocab_size=6251,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1408,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    print("\n" + "=" * 60)
    print("Financial Foundation Model — Decoder-Only Pretraining (HF)")
    print("=" * 60)
    print(f"  Architecture:    Llama (decoder-only)")
    print(f"  Hidden size:     {config.hidden_size}")
    print(f"  Layers:          {config.num_hidden_layers}")
    print(f"  Vocab size:      {config.vocab_size}")
    print(f"  Device:          {device_str}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Batch size:      {args.per_device_train_batch_size}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 60 + "\n")

    model = LlamaForCausalLM(config)

    from src.clm_data import load_corpus_and_tokenize

    train_dataset = load_corpus_and_tokenize(
        data_path=args.train_data,
        merchant_hash_size=args.merchant_hash_size,
        seq_length=args.seq_length,
    )

    eval_dataset = None
    if args.val_data:
        eval_dataset = load_corpus_and_tokenize(
            data_path=args.val_data,
            merchant_hash_size=args.merchant_hash_size,
            seq_length=args.seq_length,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        fp16=args.fp16 and device_str == "cuda",
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    class DataCollatorForCLM:
        def __call__(self, features):
            import torch
            input_ids = torch.stack([f["input_ids"] for f in features])
            labels = torch.stack([f["labels"] for f in features])
            return {"input_ids": input_ids, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCLM(),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
