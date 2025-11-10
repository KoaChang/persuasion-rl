# Quick Start Guide

This guide gets you from zero to training in 30 minutes.

## TL;DR

**Goal**: Train Qwen2.5-0.5B on persuasion data using AWS GPU

**Time**: 8-15 hours of training (mostly hands-off)

**Cost**: ~$8-16

**Dataset Size**: 30,000-60,000 SFT examples (recommended: 50,000)

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
✅ **Training Data**: 30k-60k persuasion examples (train/val/test splits)  
✅ **AI Preference Pool**: 10,000 prompts × 2 responses (for RLAIF)  
✅ **Human Preference Pool**: 300 prompts × 2 responses (for RLHF)  
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

## Recommended Dataset Size

Based on your RL plan (300 RLHF, 10k RLAIF):

**Target: 50,000 SFT examples**

This gives you:
- 160x your RLHF data
- 5x your RLAIF data
- Strong foundation for RL improvements

See **[SFT_DATASET_SIZE_ANALYSIS.md](SFT_DATASET_SIZE_ANALYSIS.md)** for detailed analysis.

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

| Item | Cost |
|------|------|
| g5.xlarge (12 hours) | ~$12 |
| Storage (100GB, 1 week) | ~$2.50 |
| Data transfer | ~$0.20 |
| **Total** | **~$15** |

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
- **Dataset Sizing**: [SFT_DATASET_SIZE_ANALYSIS.md](SFT_DATASET_SIZE_ANALYSIS.md) - Why 50k?
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

