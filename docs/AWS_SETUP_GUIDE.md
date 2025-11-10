# AWS Setup Guide for SFT Training

This guide walks you through setting up AWS and running the complete SFT training pipeline (50k examples, 10-12 hours, ~$13-16).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [Request GPU Quota](#request-gpu-quota)
4. [Create SSH Key](#create-ssh-key)
5. [Launch EC2 Instance](#launch-ec2-instance)
6. [Connect and Run Pipeline](#connect-and-run-pipeline)
7. [Monitor Training](#monitor-training)
8. [Download Results](#download-results)
9. [Stop Instance](#stop-instance)

---

## Prerequisites

- [ ] AWS account (credit card required)
- [ ] GitHub repository pushed with your code
- [ ] Weights & Biases account ([wandb.ai](https://wandb.ai) - free)

**Cost**: ~$13-16 for full pipeline (10-12 hours on g5.xlarge)

---

## AWS Account Setup

1. **Create account** at [aws.amazon.com](https://aws.amazon.com) → "Create an AWS Account"
2. **Fill in details**: Email, password, credit card, phone verification
3. **Choose plan**: Basic Support (Free)
4. **Sign in** at [console.aws.amazon.com](https://console.aws.amazon.com)

---

## Request GPU Quota

New AWS accounts have 0 GPU quota by default. You must request access:

1. **Search** "Service Quotas" in AWS Console
2. **Click** AWS services → "Amazon Elastic Compute Cloud (Amazon EC2)"
3. **Search** "Running On-Demand G and VT instances"
4. **Click** the quota → "Request quota increase"
5. **Enter**: New quota = **4 vCPUs**
6. **Reason**: "CS230 deep learning project - training persuasion model with LoRA"
7. **Submit** and wait for email approval (15 minutes - 24 hours)

⚠️ **Wait for approval before proceeding!**

---

## Create SSH Key

1. **Search** "EC2" in AWS Console → EC2 Dashboard
2. **Left sidebar** → Network & Security → "Key Pairs"
3. **Click** "Create key pair"
4. **Name**: `persuasion-rl-key`
5. **Type**: RSA
6. **Format**: `.pem` (macOS/Linux)
7. **Click** "Create" - key downloads automatically

**Set permissions** (macOS):

```bash
mkdir -p ~/.ssh
mv ~/Downloads/persuasion-rl-key.pem ~/.ssh/
chmod 400 ~/.ssh/persuasion-rl-key.pem
```

---

## Launch EC2 Instance

1. **EC2 Dashboard** → "Launch Instance"

2. **Name**: `persuasion-rl-training`

3. **AMI**: Browse more AMIs → Search "Deep Learning AMI GPU PyTorch" → Select PyTorch 2.0.1 (Ubuntu 20.04)

4. **Instance type**: `g5.xlarge` (4 vCPUs, 16GB RAM, 1x A10G GPU 24GB)

5. **Key pair**: Select `persuasion-rl-key`

6. **Network**: Click "Edit"

   - Auto-assign public IP: Enable
   - Firewall: Create security group
   - Name: `persuasion-rl-sg`
   - Inbound rules: SSH from "My IP"

7. **Storage**: Change to **100 GiB** (gp3)

8. **Advanced details** (expand):

   - Shutdown behavior: "Stop"
   - Termination protection: Enable

9. **Click** "Launch instance"

10. **Wait** for status: Pending → Running → 2/2 checks passed (~3-5 minutes)

11. **Note your Public IPv4 address** (you'll need it to connect)

---

## Connect and Run Pipeline

### Connect to Instance

```bash
# From your Mac Terminal
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP
# Type "yes" if asked about host authenticity
```

### Verify GPU

```bash
nvidia-smi
# Should show: NVIDIA A10G GPU with 23GB VRAM
```

### Clone and Setup

```bash
cd ~
git clone https://github.com/YOUR-USERNAME/persuasion-rl.git
cd persuasion-rl
bash scripts/setup_env.sh
source venv/bin/activate
```

### Login to W&B

```bash
wandb login
# Get API key from: https://wandb.ai/authorize
# Paste and press Enter
```

### Run Complete Pipeline

```bash
# Use tmux to keep training running if connection drops
tmux new -s training

# Run full pipeline (50k SFT + preference generation)
bash scripts/run_sft_aws.sh

# Detach from tmux: Ctrl+B then D
# Reattach later: tmux attach -t training
```

**⏱️ Time**: 13-16 hours  
**💰 Cost**: ~$13-16  
**Output**: Trained model + 8k RLAIF + 300 RLHF preference pairs

---

## Monitor Training

### Weights & Biases Dashboard

1. Go to [wandb.ai](https://wandb.ai) → your project "persuasion-rl-cs230"
2. Click on run "qwen-sft-baseline"
3. Watch:
   - **Loss**: Should decrease steadily
   - **GPU Util**: Should be ~95%+
   - **Time remaining**: Shows ETA

### Check from Terminal

```bash
# Reattach to tmux if disconnected
tmux attach -t training

# Or SSH in a new window and check GPU
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP
watch -n 1 nvidia-smi
```

---

## Download Results

After training completes, download from your **local Mac** (not SSH):

```bash
# Create results directory
mkdir -p ~/Downloads/persuasion-rl-results

# Download trained model (~1-2GB, takes 5-15 minutes)
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final \
    ~/Downloads/persuasion-rl-results/model/

# Download preference data
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/data/preferences \
    ~/Downloads/persuasion-rl-results/
```

---

## Stop Instance

⚠️ **IMPORTANT**: Stop when done to avoid charges!

1. **Download all results first!** (see above)
2. **EC2 Dashboard** → Instances
3. **Select** your instance
4. **Instance state** → "Stop instance"

To terminate completely (deletes everything):

- Disable termination protection first
- Instance state → "Terminate instance"

**Check costs**: [AWS Billing Dashboard](https://console.aws.amazon.com/billing/)

---

## Quick Reference

```bash
# Connect
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP

# Setup and run (first time)
cd ~ && git clone YOUR-REPO-URL
cd persuasion-rl
bash scripts/setup_env.sh
source venv/bin/activate
wandb login
tmux new -s training
bash scripts/run_sft_aws.sh

# Check status (if disconnected)
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP
tmux attach -t training

# Download results (from Mac)
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final \
    ~/Downloads/persuasion-rl-results/
```

---

**🎉 That's it! Full pipeline runs automatically for ~13-16 hours.**

**Monitor at**: [wandb.ai](https://wandb.ai)  
**Expected cost**: ~$13-16  
**Output**: Trained model + 8k RLAIF pairs + 300 RLHF pairs

---

**🎉 You're ready to train! Good luck with your CS230 project!**
