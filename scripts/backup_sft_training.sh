#!/bin/bash
# Backup SFT Training Results
# Creates a compressed archive of the SFT training outputs

set -e

cd /home/ubuntu/persuasion-rl

echo "================================================"
echo "Backing up SFT Training Results"
echo "================================================"

# Create timestamped backup name
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="sft_training_backup_${BACKUP_DATE}"

echo "Backup name: ${BACKUP_NAME}.tar.gz"
echo ""

# Check if the model directory exists
if [ ! -d "models/checkpoints/qwen-sft" ]; then
    echo "Error: SFT model directory not found!"
    exit 1
fi

echo "Creating compressed backup..."
echo "This may take a few minutes..."
echo ""

# Create compressed backup
# Excludes large optimizer states to reduce size
tar -czf ${BACKUP_NAME}.tar.gz \
  models/checkpoints/qwen-sft/ \
  configs/sft_config.yaml \
  wandb/run-20251112_071257-4ums0qx9/ \
  --exclude="optimizer.pt" \
  --exclude="scheduler.pt" \
  --exclude="scaler.pt" \
  2>/dev/null || true

# If wandb directory doesn't exist, create backup without it
if [ ! -f "${BACKUP_NAME}.tar.gz" ]; then
    echo "Note: Creating backup without wandb logs..."
    tar -czf ${BACKUP_NAME}.tar.gz \
      models/checkpoints/qwen-sft/ \
      configs/sft_config.yaml
fi

echo "================================================"
echo "Backup Complete!"
echo "================================================"
echo ""
echo "Backup file: ${BACKUP_NAME}.tar.gz"
echo "Location: /home/ubuntu/persuasion-rl/${BACKUP_NAME}.tar.gz"
echo ""

# Show backup size
BACKUP_SIZE=$(ls -lh ${BACKUP_NAME}.tar.gz | awk '{print $5}')
echo "Backup size: ${BACKUP_SIZE}"
echo ""

# Show contents summary
echo "Backup contains:"
tar -tzf ${BACKUP_NAME}.tar.gz | head -20
echo "... (and more files)"
echo ""

echo "To transfer to your Mac, run this command on your local machine:"
echo "================================================"
echo "scp -i /path/to/YOUR_KEY.pem ubuntu@YOUR_AWS_IP:/home/ubuntu/persuasion-rl/${BACKUP_NAME}.tar.gz ~/Desktop/"
echo "================================================"

