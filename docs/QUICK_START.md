# Quick Start Guide

This guide gets you from zero to training in 30 minutes.

## TL;DR

**Goal**: Train Qwen2.5-0.5B on persuasion data using AWS GPU

**Time**: 4-6 hours of training (mostly hands-off)

**Cost**: ~$4-6

**Dataset Size**: 11,750 total examples from CMV (9,400 SFT training, 2,150 RLAIF, 200 RLHF, 115 final eval)

## Prerequisites Checklist

Before you start:

- [ ] AWS account with credit card
- [ ] Git repository pushed to GitHub
- [ ] Weights & Biases account (free at wandb.ai)
- [ ] SSH client (Terminal on Mac)

## Three-Step Process

### Step 1: AWS Setup (30 minutes)

Follow **[AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md)** to:

1. Create AWS account
2. Request GPU quota increase (g5.xlarge)
3. Create SSH key pair
4. Launch EC2 instance
5. Connect via SSH

**Key decisions**:

- **Instance**: g5.xlarge ($1/hour, recommended) or g4dn.xlarge ($0.50/hour, slower)
- **Storage**: 100GB
- **Region**: us-east-1 (usually cheapest)

### Step 2: Run Training (8-15 hours, automated)

SSH into your instance, then:

```bash
# Clone your repository
git clone YOUR-REPO-URL
cd persuasion-rl

# Setup environment
bash scripts/setup_env.sh
source venv/bin/activate

# Login to W&B
wandb login

# Run full pipeline (automated)
bash scripts/run_sft_aws.sh
```

This single command will:

1. Download CMV and PersuasionForGood datasets
2. Preprocess both datasets
3. Create SFT training set
4. Train model with LoRA
5. Generate preference data for RLHF/RLAIF

**Monitor progress**: Check [wandb.ai](https://wandb.ai) dashboard

### Step 3: Download Results & Stop Instance (15 minutes)

From your **local Mac**:

```bash
# Download trained model and preference data
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final \
    ~/Downloads/persuasion-rl-results/

scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/data/preferences \
    ~/Downloads/persuasion-rl-results/
```

**Then STOP the instance** (in AWS Console):

- EC2 → Instances → Select your instance → Instance state → Stop

## What You'll Get

After completion:

✅ **Trained SFT Model**: Qwen2.5-0.5B + LoRA adapters  
✅ **Training Data**: 11,750 CMV examples (9,400 train / 1,175 val / 1,175 test)  
✅ **AI Preference Pool**: 2,150 prompts × 2 responses (for RLAIF)  
✅ **Human Preference Pool**: 200 prompts × 2 responses (for RLHF)  
✅ **Final Eval Set**: 115 prompts (completely held-out)  
✅ **Training Metrics**: Loss curves, GPU utilization in W&B  
✅ **Evaluation Samples**: Model outputs on test set

## Key Configuration

Default settings (in `configs/sft_config.yaml`):

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
lora_r: 8
learning_rate: 1e-4
epochs: 3
batch_size: 64 (effective)
max_seq_length: 1024
```

## Default Dataset Configuration

**11,750 total examples from CMV dataset:**

- **9,400** (80%) → SFT training
- **1,175** (10%) → SFT validation
- **1,175** (10%) → SFT test

**Preference generation from val+test (2,350 prompts):**

- **2,150** prompts → RLAIF pool (AI-graded preferences) - 91.5%
- **200** prompts → RLHF pool (human-graded preferences) - 8.5%

**Final evaluation:**
- **115** prompts → Completely held-out (from 11,865 - 11,750 reserve)

**Ratios:**

- SFT:RLAIF = 9.4k:2.15k = 4.37x (good, close to 5-10x optimal)
- SFT:RLHF = 9.4k:200 = 47x (reasonable for project scope)
- RLAIF:RLHF = 2.15k:200 = 10.75x (excellent for AI warmup)

See **[DATASET_SIZES_SUMMARY.md](DATASET_SIZES_SUMMARY.md)** for detailed analysis.

## Common Issues

### GPU quota denied

**Solution**: Wait for approval (15min-24hrs) or try g4dn.xlarge

### Out of memory during training

**Solution**: Reduce batch size or use `--use-4bit` flag

### Training too slow

**Solution**: Verify GPU usage with `nvidia-smi`, ensure fp16=true

### Can't SSH into instance

**Solution**: Check key permissions (chmod 400), verify security group allows SSH from your IP

## Cost Breakdown

| Item                    | Cost     |
| ----------------------- | -------- |
| g5.xlarge (12 hours)    | ~$12     |
| Storage (100GB, 1 week) | ~$2.50   |
| Data transfer           | ~$0.20   |
| **Total**               | **~$15** |

**Save money**:

- Use g4dn.xlarge instead (~$10 total)
- Reduce dataset to 30k examples (~$8 total)
- Terminate (not just stop) when done

## Next Steps After SFT

Once SFT is complete, you'll move to:

1. **RLAIF**: Use GPT-4 to label AI pool → Train DPO
2. **RLHF**: Manually label human pool → Train DPO
3. **Evaluation**: Compare all models

## Need Help?

- **AWS Setup**: [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md) - Complete walkthrough
- **Execution Details**: [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - Step-by-step commands
- **Dataset Sizing**: [DATASET_SIZES_SUMMARY.md](DATASET_SIZES_SUMMARY.md) - Why 11.75k?
- **Progress Tracking**: [TODO.md](TODO.md) - What's next?

## Quick Reference

```bash
# Connect to AWS
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-IP

# Check GPU
nvidia-smi

# Run training
cd ~/persuasion-rl
source venv/bin/activate
bash scripts/run_sft_aws.sh

# Monitor (separate window)
watch -n 1 nvidia-smi

# Download results (from local Mac)
scp -i ~/.ssh/persuasion-rl-key.pem -r ubuntu@YOUR-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final ~/Downloads/

# Stop instance (in AWS console)
EC2 → Instances → Stop
```

---

**Ready to start? Go to [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md)!**
