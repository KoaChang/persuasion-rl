"""
Train SFT model with LoRA on persuasion datasets.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.model_utils import load_tokenizer, load_base_model, print_trainable_parameters


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize the full_text field for supervised fine-tuning.
    The full_text includes both the prompt and the response.
    """
    # Tokenize
    result = tokenizer(
        examples["full_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    
    return result


def create_datasets(config: Dict, tokenizer):
    """Load and preprocess train/val datasets."""
    train_file = config["data"]["train_file"]
    val_file = config["data"]["val_file"]
    
    print(f"Loading training data from {train_file}...")
    train_data = load_jsonl(train_file)
    print(f"  Loaded {len(train_data)} training examples")
    
    print(f"Loading validation data from {val_file}...")
    val_data = load_jsonl(val_file)
    print(f"  Loaded {len(val_data)} validation examples")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize
    max_length = config["model"]["max_seq_length"]
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=False,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val"
    )
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train SFT model with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(config.get("seed", 42))
    
    # Initialize wandb
    if config.get("wandb"):
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=config["wandb"]["run_name"],
            config=config
        )
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config['model']['base_model']}...")
    tokenizer = load_tokenizer(
        config["model"]["base_model"],
        trust_remote_code=config["model"].get("trust_remote_code", True)
    )
    
    # Load base model
    print(f"Loading base model: {config['model']['base_model']}...")
    model = load_base_model(
        config["model"]["base_model"],
        use_4bit=args.use_4bit,
        trust_remote_code=config["model"].get("trust_remote_code", True)
    )
    
    # Apply LoRA
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"].get("bias", "none"),
        task_type=config["lora"].get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Load datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    
    # Training arguments
    training_config = config["training"]
    output_dir = training_config["output_dir"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_config.get("fp16", True) and torch.cuda.is_available(),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "adamw_torch"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        report_to="wandb" if config.get("wandb") else "none",
        run_name=training_config.get("run_name", "sft-training"),
        seed=config.get("seed", 42),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(output_dir, "final")
    print(f"\nSaving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")
    
    if config.get("wandb"):
        wandb.finish()


if __name__ == "__main__":
    main()

