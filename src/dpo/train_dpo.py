#!/usr/bin/env python3
"""
DPO Training Script using TRL library.

This script trains models using Direct Preference Optimization (DPO) for
three different stages:
  1. rlaif: SFT model → DPO on AI preferences (2,150 examples)
  2. rlhf: SFT model → DPO on human preferences (200 examples)
  3. rlaif_to_rlhf: RLAIF model → DPO on human preferences

Usage:
    python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlaif
    python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlhf
    python src/dpo/train_dpo.py --config configs/dpo_config.yaml --stage rlaif_to_rlhf
"""

import argparse
import os
import sys
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_utils import load_base_model, print_trainable_parameters


def load_preference_dataset(train_file, val_file):
    """Load preference dataset from JSONL files."""
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file}
    )
    return dataset


def get_reference_model(config):
    """
    Load reference model (frozen) for DPO.

    For all stages, we use SFT as the reference model (Option A).
    This keeps all DPO objectives pulling toward the same anchor.
    """
    print("\nLoading reference model (SFT)...")

    # Load base model
    base_model = load_base_model(
        config["base_model"],
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load SFT LoRA adapters
    ref_model = PeftModel.from_pretrained(
        base_model,
        config["sft_model_path"]
    )

    # Merge LoRA into base for reference model (makes it simpler)
    ref_model = ref_model.merge_and_unload()

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    print("Reference model loaded and frozen")
    return ref_model


def get_policy_model(config, stage_config, stage):
    """
    Load policy model (trainable) for DPO.

    Args:
        config: Full config dict
        stage_config: Config for specific stage (rlaif/rlhf/rlaif_to_rlhf)
        stage: Stage name

    Returns:
        Policy model with new LoRA adapters for DPO training
    """
    print(f"\nLoading policy model for {stage.upper()} stage...")

    # Load base model
    base_model = load_base_model(
        config["base_model"],
        device_map="auto",
        torch_dtype=torch.float16
    )

    if stage in ["rlaif", "rlhf"]:
        # Start from SFT model
        print("Loading SFT LoRA adapters...")
        policy_model = PeftModel.from_pretrained(
            base_model,
            config["sft_model_path"]
        )

        # Merge SFT LoRA into base (so we can apply new LoRA for DPO)
        print("Merging SFT LoRA adapters into base model...")
        policy_model = policy_model.merge_and_unload()

    elif stage == "rlaif_to_rlhf":
        # Start from RLAIF model
        print("Loading RLAIF LoRA adapters...")
        policy_model = PeftModel.from_pretrained(
            base_model,
            stage_config["init_model_path"]
        )

        # Merge RLAIF LoRA into base
        print("Merging RLAIF LoRA adapters into base model...")
        policy_model = policy_model.merge_and_unload()

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Apply new LoRA for DPO training
    print(f"Applying new LoRA adapters for {stage.upper()} DPO training...")
    lora_config = LoraConfig(**stage_config["lora"])
    policy_model = get_peft_model(policy_model, lora_config)

    print_trainable_parameters(policy_model)

    return policy_model


def main():
    parser = argparse.ArgumentParser(description="Train DPO models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to DPO config file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["rlaif", "rlhf", "rlaif_to_rlhf"],
        help="Training stage"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    stage_config = config[args.stage]
    dpo_config = config["dpo"]
    training_config = config["training"]

    # Create output directory
    os.makedirs(stage_config["output_dir"], exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    print(f"\nLoading preference datasets...")
    print(f"  Train: {stage_config['train_file']}")
    print(f"  Val: {stage_config['val_file']}")
    dataset = load_preference_dataset(
        stage_config["train_file"],
        stage_config["val_file"]
    )
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Val examples: {len(dataset['validation'])}")

    # Load models
    policy_model = get_policy_model(config, stage_config, args.stage)
    reference_model = get_reference_model(config)

    # DPO training arguments (using DPOConfig which extends TrainingArguments)
    print("\nSetting up training arguments...")
    training_args = DPOConfig(
        output_dir=stage_config["output_dir"],
        learning_rate=stage_config["learning_rate"],
        num_train_epochs=stage_config["num_epochs"],
        per_device_train_batch_size=stage_config["per_device_train_batch_size"],
        gradient_accumulation_steps=stage_config["gradient_accumulation_steps"],
        warmup_ratio=stage_config["warmup_ratio"],
        weight_decay=stage_config["weight_decay"],
        max_grad_norm=stage_config["max_grad_norm"],
        fp16=training_config["fp16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        logging_steps=training_config["logging_steps"],
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        report_to="wandb" if config["wandb"]["enabled"] else "none",
        run_name=f"{config['wandb']['project']}-{args.stage}" if config["wandb"]["enabled"] else None,
        remove_unused_columns=False,
        # DPO-specific settings
        beta=dpo_config["beta"],
        max_length=dpo_config["max_seq_length"],
        max_prompt_length=dpo_config["max_prompt_length"],
        max_completion_length=dpo_config["max_target_length"],
    )

    # Print training summary
    effective_batch_size = (
        stage_config["per_device_train_batch_size"] *
        stage_config["gradient_accumulation_steps"]
    )
    total_steps = (
        len(dataset["train"]) * stage_config["num_epochs"] // effective_batch_size
    )

    print("\n" + "=" * 80)
    print(f"DPO TRAINING SETUP: {args.stage.upper()}")
    print("=" * 80)
    print(f"Stage: {args.stage}")
    print(f"Output directory: {stage_config['output_dir']}")
    print(f"\nDataset:")
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Val examples: {len(dataset['validation'])}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {stage_config['learning_rate']}")
    print(f"  Epochs: {stage_config['num_epochs']}")
    print(f"  Per-device batch size: {stage_config['per_device_train_batch_size']}")
    print(f"  Gradient accumulation steps: {stage_config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Total training steps: ~{total_steps}")
    print(f"\nDPO settings:")
    print(f"  Beta: {dpo_config['beta']}")
    print(f"  Max sequence length: {dpo_config['max_seq_length']}")
    print(f"  Max prompt length: {dpo_config['max_prompt_length']}")
    print(f"  Max completion length: {dpo_config['max_target_length']}")
    print("=" * 80)

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING DPO TRAINING")
    print("=" * 80)
    dpo_trainer.train()

    # Save final model
    print("\n" + "=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)
    final_output_dir = f"{stage_config['output_dir']}/final"
    dpo_trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"\n✓ {args.stage.upper()} training complete!")
    print(f"Model saved to: {final_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
