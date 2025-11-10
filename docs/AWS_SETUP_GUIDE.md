# Complete AWS Setup Guide for SFT Training

This guide walks you through every step of setting up AWS for GPU training, from account creation to running your first training job.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [Request GPU Instance Quota](#request-gpu-instance-quota)
4. [Create SSH Key Pair](#create-ssh-key-pair)
5. [Launch EC2 Instance](#launch-ec2-instance)
6. [Connect to Instance](#connect-to-instance)
7. [Setup Training Environment](#setup-training-environment)
8. [Run Training](#run-training)
9. [Monitor Training](#monitor-training)
10. [Download Results](#download-results)
11. [Stop/Terminate Instance](#stopterminate-instance)
12. [Cost Management](#cost-management)

---

## Prerequisites

Before you start, you'll need:

- [ ] Credit card (for AWS billing)
- [ ] Email address
- [ ] Phone number (for verification)
- [ ] GitHub repository URL (after pushing your code)
- [ ] Weights & Biases account (free at wandb.ai)

---

## AWS Account Setup

### Step 1: Create AWS Account

1. **Go to** [aws.amazon.com](https://aws.amazon.com)

2. **Click** "Create an AWS Account" (top right)

3. **Enter** your email and choose "Personal account"

4. **Fill in** your details:

   - Account name (e.g., "CS230-Persuasion-RL")
   - Email address
   - Password

5. **Add credit card** information

   - AWS requires this even for free tier
   - You won't be charged unless you use paid services

6. **Verify your phone number**

   - Choose text or voice call
   - Enter the verification code

7. **Select support plan**: Choose "Basic Support (Free)"

8. **Wait for account activation** (can take 5-15 minutes)
   - You'll receive an email when ready

### Step 2: Sign in to AWS Console

1. **Go to** [console.aws.amazon.com](https://console.aws.amazon.com)

2. **Sign in** as "Root user" with your email and password

3. **Set up MFA (Multi-Factor Authentication)** - RECOMMENDED
   - Click your name (top right) → Security credentials
   - Under "Multi-factor authentication (MFA)", click "Assign MFA device"
   - Follow the setup wizard (use Google Authenticator or similar)

---

## Request GPU Instance Quota

By default, AWS limits new accounts to 0 GPU instances. You need to request quota increase.

### Step 1: Navigate to Service Quotas

1. **In AWS Console**, search for "Service Quotas" in the top search bar

2. **Click** "Service Quotas"

### Step 2: Request GPU Quota Increase

1. **In the left sidebar**, click "AWS services"

2. **Search for** "Amazon Elastic Compute Cloud (Amazon EC2)"

3. **Click** on it to open EC2 quotas

4. **Search for** "Running On-Demand G and VT instances"

   - This covers g5.xlarge and g4dn.xlarge

5. **Click** on the quota name

6. **Click** "Request quota increase" button (top right)

7. **Enter the increase value**:

   - Current quota: 0 vCPUs
   - **New quota: 4 vCPUs** (enough for 1x g5.xlarge or g4dn.xlarge)
   - Reason: "For CS230 deep learning course project - training a persuasion model with LoRA"

8. **Click** "Request"

### Step 3: Wait for Approval

- **Typical wait time**: 15 minutes to 24 hours
- **Check status**: Service Quotas → Dashboard → Requests
- **You'll receive an email** when approved

**⚠️ IMPORTANT**: Don't proceed to instance launch until this is approved!

### Alternative: Start with CPU (Not Recommended)

If you need to test immediately:

- Launch a `t2.large` (CPU only, no quota needed)
- Test data preprocessing only
- Cost: ~$0.09/hour

---

## Create SSH Key Pair

You need an SSH key to securely connect to your instance.

### Step 1: Navigate to EC2

1. **In AWS Console**, search for "EC2" in the top search bar

2. **Click** "EC2" to open the EC2 Dashboard

### Step 2: Create Key Pair

1. **In the left sidebar**, scroll down to "Network & Security"

2. **Click** "Key Pairs"

3. **Click** "Create key pair" button (orange, top right)

4. **Configure the key pair**:

   - **Name**: `persuasion-rl-key` (or your choice)
   - **Key pair type**: RSA
   - **Private key file format**:
     - **If macOS/Linux**: Choose `.pem`
     - **If Windows**: Choose `.ppk` (for PuTTY) or `.pem` (for Git Bash)

5. **Click** "Create key pair"

6. **The key will automatically download**
   - Save it securely (you can't download it again!)
   - Suggested location: `~/Downloads/persuasion-rl-key.pem`

### Step 3: Set Correct Permissions (macOS/Linux)

```bash
# Move key to a permanent location
mkdir -p ~/.ssh
mv ~/Downloads/persuasion-rl-key.pem ~/.ssh/

# Set restrictive permissions (required by SSH)
chmod 400 ~/.ssh/persuasion-rl-key.pem

# Verify
ls -l ~/.ssh/persuasion-rl-key.pem
# Should show: -r-------- (read-only for owner)
```

---

## Launch EC2 Instance

### Step 1: Go to EC2 Launch Wizard

1. **In EC2 Dashboard**, click "Launch Instance" (orange button)

### Step 2: Configure Instance

#### 2.1 Name and Tags

- **Name**: `persuasion-rl-training`

#### 2.2 Choose AMI (Amazon Machine Image)

1. **Click** "Browse more AMIs"

2. **Search for**: `Deep Learning AMI GPU PyTorch`

3. **Select**: "Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04)"
   - ✅ Has CUDA pre-installed
   - ✅ Has PyTorch pre-installed
   - ✅ Has conda environments ready

#### 2.3 Choose Instance Type

1. **Click** "Compare instance types" or use the dropdown

2. **Search for**: `g5.xlarge`

3. **Select** `g5.xlarge`:
   - 4 vCPUs
   - 16 GB RAM
   - 1x NVIDIA A10G GPU (24GB VRAM)
   - Cost: ~$1.006/hour

**Alternative** (if g5.xlarge not available):

- **g4dn.xlarge**: 1x T4 GPU (16GB), $0.526/hour
- **g5.2xlarge**: 1x A10G GPU (24GB), 8 vCPUs, ~$1.212/hour (more RAM)

#### 2.4 Configure Key Pair

- **Select**: `persuasion-rl-key` (the one you created earlier)

#### 2.5 Network Settings

**Click "Edit"** to modify:

1. **Auto-assign public IP**: Enable (should be default)

2. **Firewall (Security Group)**:
   - Select "Create security group"
   - **Name**: `persuasion-rl-sg`
   - **Description**: "SSH access for ML training"
   - **Inbound rules**:
     - ✅ SSH (port 22) from "My IP" (should be auto-selected)
     - This restricts SSH to only your current IP address

#### 2.6 Configure Storage

- **Change** from 8 GiB to **100 GiB**
  - Datasets + model checkpoints can be large
  - gp3 (default) is fine
  - 100 GiB = ~$10/month (only while instance exists)

#### 2.7 Advanced Details (Optional but Recommended)

**Expand "Advanced details"**:

1. **Spot instances**: ❌ Don't use (can be terminated mid-training)

2. **Shutdown behavior**:

   - Change to "Stop" (safer than terminate)

3. **Termination protection**: Enable (prevents accidental deletion)

### Step 3: Review and Launch

1. **Review** your configuration in the "Summary" panel (right side):

   - AMI: Deep Learning AMI GPU PyTorch
   - Instance type: g5.xlarge
   - Key pair: persuasion-rl-key
   - Storage: 100 GiB

2. **Check** estimated costs:

   - Should show ~$1.01/hour
   - Don't worry about the monthly estimate (you won't run 24/7)

3. **Click** "Launch instance" (orange button, bottom right)

### Step 4: Wait for Instance to Start

1. **Click** "View all instances" (or go to EC2 Dashboard → Instances)

2. **Wait** for "Instance state" to show "Running" (takes 1-2 minutes)

   - Status will show: Pending → Running

3. **Wait** for "Status check" to show "2/2 checks passed" (takes 2-3 minutes)

   - This ensures the instance is fully ready

4. **Note down** your instance details:
   - **Instance ID**: `i-0123456789abcdef0` (example)
   - **Public IPv4 address**: `54.123.45.67` (example)
   - You'll need the IP address to connect

---

## Connect to Instance

### Step 1: Get Connection Command

1. **In EC2 Dashboard → Instances**, select your instance

2. **Click** "Connect" button (top right)

3. **Go to** "SSH client" tab

4. **Copy** the example command (looks like):
   ```bash
   ssh -i "~/.ssh/persuasion-rl-key.pem" ubuntu@54.123.45.67
   ```

### Step 2: Connect from Your Mac

Open Terminal and run:

```bash
# Test connection
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP

# If you get "Host key verification" warning, type "yes"
```

**Replace `YOUR-INSTANCE-IP`** with your actual IP address (from Step 1.4 above)

### Step 3: Verify GPU

Once connected, run:

```bash
nvidia-smi
```

**Expected output**:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
|  0%   25C    P0    55W / 300W |      0MiB / 23028MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✅ **You should see "NVIDIA A10G" or "Tesla T4"**

❌ **If you see an error**: Your instance doesn't have a GPU - check instance type

---

## Setup Training Environment

### Step 1: Clone Your Repository

```bash
# Navigate to home directory
cd ~

# Clone your repo (replace with your actual URL)
git clone https://github.com/YOUR-USERNAME/persuasion-rl.git

# Enter directory
cd persuasion-rl

# Verify structure
ls -la
```

### Step 2: Setup Environment

```bash
# Run setup script
bash scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Expected output**:

```
PyTorch: 2.0.1
CUDA available: True
Transformers: 4.35.0
```

### Step 3: Login to Weights & Biases

```bash
wandb login
```

**You'll be prompted** for your API key:

1. Go to [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key
3. Paste it in the terminal (it won't show as you type - that's normal)
4. Press Enter

**Verification**:

```bash
wandb verify
# Should show: "Successfully logged in"
```

---

## Run Training

### Option A: Full Automated Pipeline (Recommended)

Run everything with one command:

```bash
# This will:
# 1. Download datasets
# 2. Preprocess CMV and P4G
# 3. Create SFT dataset
# 4. Train model
# 5. Generate preference data

bash scripts/run_sft_aws.sh
```

**⏱️ Estimated time**: 8-15 hours total

**💰 Estimated cost**: $8-$16 (at $1/hour)

**📊 Monitor**: Check Weights & Biases dashboard at wandb.ai

### Option B: Step-by-Step Manual (For More Control)

#### 1. Download Datasets (~20 minutes)

```bash
python src/data/download_datasets.py --output-dir data/raw
```

#### 2. Preprocess CMV (~15 minutes)

```bash
python src/data/preprocess_cmv.py \
    --input-dir data/raw/cmv \
    --output-file data/processed/cmv_examples.jsonl
```

#### 3. Preprocess PersuasionForGood (~15 minutes)

```bash
python src/data/preprocess_persuasionforgood.py \
    --input-dir data/raw/persuasionforgood \
    --output-file data/processed/p4g_examples.jsonl
```

#### 4. Check Dataset Sizes

```bash
wc -l data/processed/cmv_examples.jsonl
wc -l data/processed/p4g_examples.jsonl
```

**Aim for**: 30,000-60,000 total examples (see `docs/SFT_DATASET_SIZE_ANALYSIS.md`)

#### 5. Create SFT Dataset (~5 minutes)

```bash
python src/data/create_sft_dataset.py \
    --cmv-file data/processed/cmv_examples.jsonl \
    --p4g-file data/processed/p4g_examples.jsonl \
    --output-dir data/processed \
    --max-examples 50000  # Optional: limit to 50k if you have more
```

#### 6. Train SFT Model (~8-12 hours)

```bash
python src/sft/train_sft.py --config configs/sft_config.yaml
```

**This is the long step!** See "Monitor Training" section below.

#### 7. Generate Preference Data (~2-4 hours)

```bash
python src/sft/generate_preferences.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/sft_test.jsonl \
    --output-dir data/preferences \
    --ai-pool-size 10000 \
    --human-pool-size 300
```

#### 8. Evaluate Model (~5 minutes)

```bash
python src/eval/evaluate_model.py \
    --model-path models/checkpoints/qwen-sft/final \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --test-file data/processed/sft_test.jsonl \
    --num-samples 10 \
    --output-file models/checkpoints/qwen-sft/evaluation.json
```

---

## Monitor Training

### Option 1: Weights & Biases Dashboard (Best)

1. **Go to** [wandb.ai/YOUR-USERNAME/persuasion-rl-cs230](https://wandb.ai)

2. **Click** on your run (e.g., "qwen-sft-baseline")

3. **Monitor these metrics**:

   - **Loss**: Should decrease steadily
   - **Learning rate**: Should follow cosine schedule
   - **GPU Memory**: Should be ~18-20GB for g5.xlarge
   - **System/GPU**: Should show ~95%+ utilization

4. **Set up alerts** (optional):
   - Click "Alerts" → "New Alert"
   - Alert if: `train/loss > 2.0` after 1 hour (indicates problem)

### Option 2: Watch Training Logs

In your SSH session:

```bash
# Watch real-time logs
tail -f models/checkpoints/qwen-sft/trainer_state.json

# Or use tmux to keep session alive (recommended)
tmux new -s training
# Run training inside tmux
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

### Option 3: Monitor GPU

In a separate terminal window:

```bash
# SSH in again
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP

# Watch GPU utilization
watch -n 1 nvidia-smi
```

**What to look for**:

- GPU Utilization: Should be 90-100%
- GPU Memory: ~18-20GB / 23GB used
- Temperature: 60-80°C (normal for training)
- Power: 200-280W / 300W

---

## Download Results

After training completes:

### Option 1: SCP (Secure Copy)

From your **local Mac terminal** (not SSH):

```bash
# Create results directory
mkdir -p ~/Downloads/persuasion-rl-results

# Download trained model
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final \
    ~/Downloads/persuasion-rl-results/model/

# Download preference data
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/data/preferences \
    ~/Downloads/persuasion-rl-results/

# Download evaluation results
scp -i ~/.ssh/persuasion-rl-key.pem \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/evaluation.json \
    ~/Downloads/persuasion-rl-results/
```

**⏱️ Transfer time**: 5-15 minutes (model is ~1-2GB)

### Option 2: AWS S3 (Better for large files)

On your **AWS instance**:

```bash
# Install AWS CLI (if not already)
sudo apt-get update && sudo apt-get install -y awscli

# Upload to S3
aws s3 mb s3://your-bucket-name-persuasion-rl  # Create bucket
aws s3 sync models/checkpoints/qwen-sft/final s3://your-bucket-name-persuasion-rl/model/
aws s3 sync data/preferences s3://your-bucket-name-persuasion-rl/preferences/
```

On your **local Mac**:

```bash
# Download from S3
aws s3 sync s3://your-bucket-name-persuasion-rl ~/Downloads/persuasion-rl-results/
```

---

## Stop/Terminate Instance

### ⚠️ IMPORTANT: Always stop/terminate when done to avoid charges!

### Option 1: Stop Instance (Pause - Recommended)

**Use this if you might need the instance again**

1. **Go to** EC2 Dashboard → Instances

2. **Select** your instance

3. **Click** "Instance state" → "Stop instance"

4. **Costs**:

   - ✅ No compute charges ($0/hour)
   - ⚠️ Still pay for storage (~$10/month for 100GB)

5. **To restart later**:
   - Select instance → "Instance state" → "Start instance"
   - **Note**: IP address will change!

### Option 2: Terminate Instance (Delete - Permanent)

**Use this when completely done with the project**

1. **⚠️ FIRST**: Download all results (see above)

2. **Disable termination protection**:

   - Select instance → Actions → Instance settings → Change termination protection → Disable

3. **Go to** EC2 Dashboard → Instances

4. **Select** your instance

5. **Click** "Instance state" → "Terminate instance"

6. **Confirm** termination

7. **Costs**:
   - ✅ No compute charges
   - ✅ No storage charges
   - Everything is deleted (cannot be recovered!)

### Check Your Costs

**Go to** [AWS Billing Dashboard](https://console.aws.amazon.com/billing/)

- **View** current month charges
- **Set up** billing alerts (recommended):
  - Billing → Budgets → Create budget
  - Set alert at $20, $50, $100

---

## Cost Management

### Expected Costs for Full Pipeline

| Item                | Cost      | Duration | Total    |
| ------------------- | --------- | -------- | -------- |
| g5.xlarge compute   | $1.01/hr  | 12 hours | ~$12     |
| EBS storage (100GB) | $10/month | 1 week   | ~$2.50   |
| Data transfer out   | ~$0.09/GB | 2GB      | ~$0.20   |
| **TOTAL**           |           |          | **~$15** |

### Cost-Saving Tips

1. **Stop instance when not in use**

   - Don't leave it running overnight if training is done
   - Stop it during preprocessing steps

2. **Use g4dn.xlarge instead**

   - Half the price ($0.526/hr vs $1.01/hr)
   - Slower GPU (T4 vs A10G)
   - Training takes ~50% longer
   - Total cost might be similar, but safer for budget

3. **Reduce dataset size**

   - Use `--max-examples 30000` instead of 50000
   - Cuts training time by ~40%

4. **Set billing alerts**

   - AWS Billing → Budgets
   - Alert at $20 (before it gets expensive)

5. **Terminate after downloading**
   - Don't just stop - terminate when completely done
   - Saves $10/month in storage

---

## Troubleshooting

### "Permission denied (publickey)"

**Problem**: Can't SSH into instance

**Solutions**:

1. Check key file permissions: `ls -l ~/.ssh/persuasion-rl-key.pem` (should be 400)
2. Verify correct username: Should be `ubuntu@` not `ec2-user@`
3. Check IP address: Make sure you're using the right public IP
4. Security group: Verify SSH (port 22) is allowed from your IP

### "Quota Exceeded" when launching

**Problem**: Can't launch g5.xlarge

**Solutions**:

1. Check quota status: Service Quotas → Dashboard → Requests
2. Request increase (see "Request GPU Instance Quota" section)
3. Try g4dn.xlarge instead (different quota limit)
4. Wait for approval (can take 24 hours)

### "CUDA out of memory"

**Problem**: Training crashes with OOM error

**Solutions**:

1. Reduce batch size in `configs/sft_config.yaml`:
   ```yaml
   per_device_train_batch_size: 2 # down from 4
   gradient_accumulation_steps: 32 # up from 16
   ```
2. Use 4-bit quantization:
   ```bash
   python src/sft/train_sft.py --config configs/sft_config.yaml --use-4bit
   ```
3. Reduce sequence length:
   ```yaml
   max_seq_length: 768 # down from 1024
   ```

### Training is very slow

**Problem**: Less than 50 steps/second

**Solutions**:

1. Check GPU usage: `nvidia-smi` should show 90%+ utilization
2. Verify fp16 is enabled: Check `fp16: true` in config
3. Check you're on GPU instance (not CPU)
4. Monitor with: `watch -n 1 nvidia-smi`

### Instance keeps disconnecting

**Problem**: SSH connection drops

**Solutions**:

1. Use tmux to persist sessions:
   ```bash
   tmux new -s training
   # Run training
   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t training
   ```
2. Add keep-alive to SSH config (`~/.ssh/config`):
   ```
   Host persuasion-rl
       HostName YOUR-INSTANCE-IP
       User ubuntu
       IdentityFile ~/.ssh/persuasion-rl-key.pem
       ServerAliveInterval 60
   ```
3. Then connect with: `ssh persuasion-rl`

---

## Next Steps After Training

Once training is complete:

1. ✅ Download all results
2. ✅ Verify model works locally (optional)
3. ✅ Stop/terminate AWS instance
4. ✅ Review training metrics in W&B
5. ✅ Analyze preference data quality
6. 📋 Plan next phases:
   - RLAIF: AI grader setup
   - RLHF: Human preference collection
   - Evaluation: Grader model

See `docs/TODO.md` for next phases!

---

## Quick Reference Commands

```bash
# Connect to instance
ssh -i ~/.ssh/persuasion-rl-key.pem ubuntu@YOUR-INSTANCE-IP

# Check GPU
nvidia-smi

# Run full pipeline
cd ~/persuasion-rl
source venv/bin/activate
bash scripts/run_sft_aws.sh

# Monitor training (in tmux)
tmux new -s training
bash scripts/run_sft_aws.sh
# Detach: Ctrl+B then D

# Reattach to training
tmux attach -t training

# Download results (from local Mac)
scp -i ~/.ssh/persuasion-rl-key.pem -r \
    ubuntu@YOUR-INSTANCE-IP:~/persuasion-rl/models/checkpoints/qwen-sft/final \
    ~/Downloads/persuasion-rl-results/
```

---

**🎉 You're ready to train! Good luck with your CS230 project!**
