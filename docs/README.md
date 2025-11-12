# Documentation Index

This folder contains comprehensive documentation for the Persuasion-RL SFT project.

## Getting Started

**New to the project?** Start here:

1. **[QUICK_START.md](QUICK_START.md)** - Get from zero to training in 30 minutes
2. **[AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md)** - Detailed AWS setup (account creation, GPU quotas, instance launch)
3. **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Step-by-step execution instructions

## Planning & Analysis

**Before training**, read these:

- **[DATASET_SIZES_SUMMARY.md](DATASET_SIZES_SUMMARY.md)** - Default configuration: 50k total (40k SFT, 8k RLAIF, 300 RLHF, 1.7k eval)
- **[TODO.md](TODO.md)** - Project roadmap and next phases

## Reference

**After implementation**:

- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - What has been built, file inventory, deliverables

## Quick Links

| Document | Purpose | Time to Read |
|----------|---------|--------------|
| [QUICK_START.md](QUICK_START.md) | Fast path to training | 5 min |
| [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md) | Complete AWS walkthrough | 20 min |
| [DATASET_SIZES_SUMMARY.md](DATASET_SIZES_SUMMARY.md) | Default configuration & ratios | 10 min |
| [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) | Detailed execution steps | 15 min |
| [TODO.md](TODO.md) | Progress & future work | 5 min |
| [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) | Implementation summary | 10 min |

## Document Descriptions

### QUICK_START.md
**TL;DR guide** - Gets you from zero to training in 3 steps:
1. AWS Setup (30 min)
2. Run Training (8-15 hours, automated)
3. Download Results (15 min)

Perfect for: First-time users who want to start immediately.

### AWS_SETUP_GUIDE.md
**Complete AWS tutorial** - Every single step from account creation to first training run:
- Creating AWS account
- Requesting GPU quotas
- Launching g5.xlarge instance
- SSH setup and connection
- Cost management
- Troubleshooting

Perfect for: Users with no AWS experience.

### DATASET_SIZES_SUMMARY.md
**Dataset configuration guide** - Default setup and ratios:
- 50,000 total examples: 40k SFT, 8k RLAIF, 300 RLHF, 1.7k eval
- Ratios: 6.25x RLAIF, 166x RLHF (optimal ranges)
- Cost breakdown (~$100-145 total project)
- Alternative configurations (30k budget, 100k maximum)
- Data flow and verification steps

Perfect for: Understanding the default configuration and alternatives.

### EXECUTION_GUIDE.md
**Detailed execution manual** - Step-by-step commands with explanations:
- Each data preprocessing step
- Training configuration options
- Preference data generation
- Monitoring strategies
- Downloading results
- Verification procedures

Perfect for: Users who want full control and understanding.

### TODO.md
**Project roadmap** - What's done, what's next:
- Completed tasks (SFT implementation)
- Pending tasks (execution on AWS)
- Future phases (RLAIF, RLHF, evaluation)

Perfect for: Tracking progress and planning ahead.

### IMPLEMENTATION_COMPLETE.md
**Implementation summary** - What has been built:
- All files created (23 total)
- Lines of code (~2,650+)
- Features implemented
- Technology stack
- Next steps

Perfect for: Understanding the codebase and deliverables.

## Navigation Tips

**If you want to...**
- **Start training now**: → [QUICK_START.md](QUICK_START.md)
- **Learn AWS from scratch**: → [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md)
- **Understand dataset sizing**: → [DATASET_SIZES_SUMMARY.md](DATASET_SIZES_SUMMARY.md)
- **See detailed commands**: → [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
- **Check what's next**: → [TODO.md](TODO.md)
- **Review what's built**: → [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

## External Documentation

Additional resources:

- **Main README**: [../README.md](../README.md) - Project overview and repository structure
- **Code Documentation**: Inline comments in all Python files
- **Configuration**: Comments in `configs/*.yaml` files
- **Scripts**: Comments in `scripts/*.sh` files

## Support

If you encounter issues:

1. Check the relevant guide above
2. Look for troubleshooting sections
3. Review error messages in logs
4. Check W&B dashboard for training metrics
5. Verify GPU with `nvidia-smi`

---

**Ready to start? Go to [QUICK_START.md](QUICK_START.md)!**
