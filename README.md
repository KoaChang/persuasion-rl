# Persuasion-RL: CS230 Deep Learning Final Project (SFT Phase)

This repository contains the Supervised Fine-Tuning (SFT) phase of a persuasion-focused reinforcement learning project. The goal is to train Qwen2.5-0.5B on persuasion dialogues from CMV (ChangeMyView) and generate preference data for future RLHF/RLAIF training.

**Default Configuration**: 50,000 total examples from CMV dataset (40,000 SFT, 8,000 RLAIF, 300 RLHF, 1,700 eval). PersuasionForGood disabled by default.

## Project Overview

This project implements the SFT portion of a larger system that will eventually include:

1. **Base Model**: Qwen2.5-0.5B (no fine-tuning)
2. **SFT Model**: Qwen2.5-0.5B + LoRA trained on persuasion dialogues (this repo)
3. **RLHF-only Model**: SFT + DPO with human preferences (future work)
4. **RLAIF→RLHF Model**: SFT + DPO with AI preferences, then human preferences (future work)

### Current Phase: SFT Training

- Train Qwen2.5-0.5B with LoRA on CMV (ChangeMyView) dataset
- Default: 50,000 total examples → 40,000 SFT training (configurable: 30k-100k)
- Generate preference data (2 responses per prompt) for future RLHF/RLAIF stages
  - RLAIF: 8,000 prompts (from val + test sets)
  - RLHF: 300 prompts (from test set)
  - Held-out eval: 1,700 prompts for final unbiased evaluation
- PersuasionForGood dataset code available but disabled by default

## Repository Structure

```
persuasion-rl/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── configs/                 # Configuration files
│   ├── sft_config.yaml      # SFT training configuration
│   └── data_config.yaml     # Data processing configuration
├── src/                     # Source code
│   ├── data/                # Data processing scripts
│   │   ├── download_datasets.py
│   │   ├── preprocess_cmv.py
│   │   ├── preprocess_persuasionforgood.py
│   │   └── create_sft_dataset.py
│   ├── sft/                 # SFT training scripts
│   │   ├── train_sft.py
│   │   └── generate_preferences.py
│   ├── models/              # Model utilities
│   │   └── model_utils.py
│   └── eval/                # Evaluation scripts
│       └── evaluate_model.py
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed datasets
│   └── preferences/         # Preference data
├── models/                  # Model directory (gitignored)
│   └── checkpoints/         # Model checkpoints
├── notebooks/               # Jupyter notebooks
│   └── exploratory_analysis.ipynb
└── scripts/                 # Utility scripts
    ├── setup_env.sh         # Environment setup
    └── run_sft_aws.sh       # Full pipeline on AWS
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training)
- AWS account (for GPU training on AWS)

### Local Setup (Mac M4 Air - for development)

```bash
# Clone the repository
git clone <your-repo-url>
cd persuasion-rl

# Run setup script
bash scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate

# Login to Weights & Biases
wandb login
```

### AWS Setup (for GPU training)

**Recommended Instance**: `g5.xlarge` (1x NVIDIA A10G GPU, 24GB VRAM, ~$1.05/hour)

Alternative: `g4dn.xlarge` (1x T4, 16GB VRAM, ~$0.50/hour)

```bash
# SSH into AWS instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repository
git clone <your-repo-url>
cd persuasion-rl

# Run full pipeline (includes setup, data processing, training, and preference generation)
bash scripts/run_sft_aws.sh
```

## Usage

### Step-by-Step Execution

#### 1. Download Datasets

```bash
python src/data/download_datasets.py --output-dir data/raw
```

This downloads:

- **CMV (ChangeMyView)**: Reddit posts where comments won deltas (changed someone's view)
- **PersuasionForGood**: Dialogues where one person persuades another to donate to charity

#### 2. Preprocess CMV

```bash
python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl \
    --min-tokens 20 \
    --max-tokens 1024
```

#### 3. Preprocess PersuasionForGood

```bash
python src/data/preprocess_persuasionforgood.py \
    --input-dir data/raw/persuasionforgood \
    --output-file data/processed/p4g_examples.jsonl \
    --min-tokens 20 \
    --max-tokens 1024
```

#### 4. Create SFT Dataset

**Default: 50k examples from CMV only** (PersuasionForGood disabled due to quality concerns)

```bash
python src/data/create_sft_dataset.py \
    --cmv-file data/processed/cmv_examples.jsonl \
    --use-cmv \
    --max-examples 50000 \
    --output-dir data/processed
```

To use different size:
```bash
# Use 30k examples (faster, cheaper)
python src/data/create_sft_dataset.py --use-cmv --max-examples 30000

# Use 100k examples (maximum quality)
python src/data/create_sft_dataset.py --use-cmv --max-examples 100000

# Use both CMV and P4G
python src/data/create_sft_dataset.py --use-cmv --use-p4g --max-examples 50000
```

This creates:
- `data/processed/sft_train.jsonl` (80%)
- `data/processed/sft_val.jsonl` (10%)
- `data/processed/sft_test.jsonl` (10%)

See [docs/CMV_ONLY_CONFIGURATION.md](docs/CMV_ONLY_CONFIGURATION.md) for details.

#### 5. Train SFT Model

**Note**: Requires GPU (CUDA)

```bash
python src/sft/train_sft.py --config configs/sft_config.yaml
```

Training hyperparameters (configured in `configs/sft_config.yaml`):

- Base model: Qwen2.5-0.5B-Instruct
- LoRA: r=8, alpha=32, dropout=0.05
- Learning rate: 1e-4
- Epochs: 3
- Effective batch size: 64
- Logging: Weights & Biases

The trained model will be saved to `models/checkpoints/qwen-sft/final/`

#### 6. Generate Preference Data

```bash
python src/sft/generate_preferences.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/sft_test.jsonl \
    --val-file data/processed/sft_val.jsonl \
    --use-val-set \
    --output-dir data/preferences \
    --ai-pool-size 8000 \
    --human-pool-size 300
```

This generates:

- **RLAIF pool**: 8,000 prompts with 2 responses each (from val + test sets)
- **RLHF pool**: 300 prompts with 2 responses each (from test set)
- **Held-out eval**: 1,700 prompts for final unbiased evaluation ⭐

Output files:

- `data/preferences/ai_pool_prompts.jsonl` - 8,000 prompts for RLAIF
- `data/preferences/ai_pool_responses.jsonl` - 8,000 × 2 responses
- `data/preferences/human_pool_prompts.jsonl` - 300 prompts for RLHF
- `data/preferences/human_pool_responses.jsonl` - 300 × 2 responses
- `data/preferences/final_eval_prompts.jsonl` - 1,700 held-out prompts for final evaluation ⭐

#### 7. Evaluate Model

```bash
python src/eval/evaluate_model.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/sft_test.jsonl \
    --num-samples 10
```

## Configuration

### SFT Training Configuration (`configs/sft_config.yaml`)

Key parameters:

- `model.base_model`: Base model from HuggingFace
- `model.max_seq_length`: Maximum sequence length (1024)
- `lora.r`: LoRA rank (8)
- `lora.lora_alpha`: LoRA alpha (32)
- `training.learning_rate`: Learning rate (1e-4)
- `training.num_epochs`: Number of epochs (3)
- `wandb.project`: W&B project name

### Data Configuration (`configs/data_config.yaml`)

Key parameters:

- `preprocessing.min_tokens`: Minimum response length (20)
- `preprocessing.max_tokens`: Maximum response length (1024)
- `preference_generation.ai_pool_size`: AI pool size (8,000)
- `preference_generation.human_pool_size`: Human pool size (300)

## Dataset Configuration

**Current Setup (Default)**:
- **Source**: CMV (ChangeMyView) only
- **Total**: 50,000 examples from CMV
- **SFT Splits**: 40k train (80%) / 5k val (10%) / 5k test (10%)
- **Preference Data** (from val+test):
  - RLAIF: 8,000 prompts (AI-graded preferences)
  - RLHF: 300 prompts (human-graded preferences)
  - Held-out eval: 1,700 prompts (final unbiased evaluation)
- **Ratios**: SFT:RLAIF = 6.25x (optimal), SFT:RLHF = 166x (optimal)

To use different configuration:
```bash
# 30k examples (faster, cheaper)
python src/data/create_sft_dataset.py --max-examples 30000

# 100k examples (maximum quality)
python src/data/create_sft_dataset.py --max-examples 100000

# Re-enable PersuasionForGood
python src/data/create_sft_dataset.py --use-cmv --use-p4g
```

See [docs/CMV_ONLY_CONFIGURATION.md](docs/CMV_ONLY_CONFIGURATION.md) for detailed guidance.

## Data Format

### SFT Training Example

```json
{
  "input_text": "[SYSTEM] You are a respectful assistant...\n[CONTEXT]\n{context}\n[TASK]\nWrite the next message...\n[RESPONSE]\n",
  "full_text": "[SYSTEM] You are a respectful assistant...\n[CONTEXT]\n{context}\n[TASK]\nWrite the next message...\n[RESPONSE]\n{response}",
  "response": "{response}",
  "metadata": {
    "source": "cmv",
    "response_tokens": 234,
    "context_tokens": 456
  }
}
```

### Preference Data Format

```json
{
  "prompt": "[SYSTEM] You are a respectful assistant...",
  "response_1": "First generated response...",
  "response_2": "Second generated response...",
  "metadata": {...}
}
```

## Next Steps (Future Work)

After completing SFT, the next phases are:

1. **RLAIF Stage**:

   - Use AI grader (GPT-4/Claude) to label AI pool preferences
   - Train DPO on AI preferences starting from SFT model

2. **RLHF Stage**:

   - Collect human labels for human pool (300 prompts)
   - Train two DPO variants:
     - RLHF-only: SFT + human DPO
     - RLAIF→RLHF: RLAIF + human DPO

3. **Evaluation**:
   - Compare all 4 models (Base, SFT, RLHF-only, RLAIF→RLHF)
   - Use grader model to rank responses
   - Analyze which approach produces more persuasive outputs

## Monitoring

Training progress is logged to Weights & Biases:

- Project: `persuasion-rl-cs230`
- Run name: `qwen-sft-baseline` (configurable)

View metrics:

- Training/validation loss
- Learning rate schedule
- GPU memory usage
- Training time

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce `per_device_train_batch_size` in config
2. Enable 4-bit quantization: `--use-4bit` flag
3. Reduce `max_seq_length` in config

### ConvoKit Download Issues

If ConvoKit downloads fail:

1. Check internet connection
2. Try downloading manually from ConvoKit website
3. Increase timeout in download script

### Model Loading Issues

If model loading fails:

1. Ensure you have enough disk space
2. Check HuggingFace token (if needed)
3. Use `trust_remote_code=True` for Qwen models

## Documentation

Comprehensive guides are available in the `docs/` folder:

- **[AWS_SETUP_GUIDE.md](docs/AWS_SETUP_GUIDE.md)** - Complete AWS setup walkthrough (account creation to first training run)
- **[DATASET_SIZES_SUMMARY.md](docs/DATASET_SIZES_SUMMARY.md)** - Default configuration: 50k total (40k SFT, 8k RLAIF, 300 RLHF, 1.7k eval)
- **[FINAL_EVALUATION_GUIDE.md](docs/FINAL_EVALUATION_GUIDE.md)** - How to use the held-out test set for unbiased evaluation (NEW)
- **[CMV_ONLY_CONFIGURATION.md](docs/CMV_ONLY_CONFIGURATION.md)** - Why CMV-only and how to customize
- **[EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md)** - Step-by-step execution instructions
- **[IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)** - Implementation summary and deliverables
- **[TODO.md](docs/TODO.md)** - Project progress and future phases

## Next Steps (Future Work)

After completing SFT, the next phases are:

1. **RLAIF Stage**:
   - Use AI grader (GPT-4/Claude) to label AI pool preferences
   - Train DPO on AI preferences starting from SFT model

2. **RLHF Stage**:
   - Collect human labels for human pool (300 prompts)
   - Train two DPO variants:
     - RLHF-only: SFT + human DPO
     - RLAIF→RLHF: RLAIF + human DPO

3. **Evaluation**:
   - Compare all 4 models (Base, SFT, RLHF-only, RLAIF→RLHF)
   - Use grader model to rank responses
   - Analyze which approach produces more persuasive outputs

## Citation

If you use this code, please cite:

```bibtex
@misc{persuasion-rl-sft,
  author = {Your Name},
  title = {Persuasion-RL: SFT Phase},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/persuasion-rl}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
