"""
Utility functions for loading and configuring models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any


def load_tokenizer(model_name: str, **kwargs):
    """Load tokenizer with appropriate settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_base_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    use_4bit: bool = False,
    trust_remote_code: bool = True,
    **kwargs
):
    """
    Load base model with optional quantization.
    
    Args:
        model_name: HuggingFace model name
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        trust_remote_code: Whether to trust remote code
    """
    if torch_dtype is None:
        torch_dtype = torch.float16
    
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs
    }
    
    if use_4bit:
        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch_dtype
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


def setup_lora(model, lora_config: Dict[str, Any]):
    """
    Apply LoRA to the model.
    
    Args:
        model: Base model
        lora_config: Dictionary with LoRA parameters
    """
    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    
    model = get_peft_model(model, peft_config)
    return model


def load_trained_model(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True
):
    """
    Load a model with trained LoRA adapters.
    
    Args:
        base_model_name: HuggingFace base model name
        adapter_path: Path to saved LoRA adapters
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
    """
    if torch_dtype is None:
        torch_dtype = torch.float16
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable %: {100 * trainable_params / all_param:.2f}%"
    )

