# SFT Project TODO

## ✅ Completed

- [x] Repository structure setup
- [x] Requirements.txt and .gitignore
- [x] Download datasets script (ConvoKit)
- [x] CMV preprocessing pipeline
- [x] PersuasionForGood preprocessing pipeline
- [x] SFT dataset creation script (with prompt template)
- [x] SFT training script with LoRA
- [x] Configuration files (YAML)
- [x] Preference data generation script
- [x] Model evaluation script
- [x] AWS setup scripts
- [x] Documentation (README, EXECUTION_GUIDE)
- [x] Exploratory analysis notebook

## 🔄 Requires Execution (Not Automatable)

These tasks require actual execution on AWS with GPU and cannot be completed programmatically:

### 1. Execute SFT Training on AWS

- **Status**: Implementation complete, awaiting execution
- **Prerequisites**: AWS g5.xlarge instance, W&B account
- **Script**: `bash scripts/run_sft_aws.sh` or manual steps
- **Estimated Time**: 8-15 hours
- **Estimated Cost**: $8-$16

### 2. Create Prompt Pools

- **Status**: Implementation complete, awaiting execution
- **Prerequisites**: Trained SFT model from step 1
- **Script**: Part of `src/sft/generate_preferences.py`
- **Output**:
  - RLAIF pool: 2,150 prompts (AI-graded preferences)
  - RLHF pool: 200 prompts (human-graded preferences)
  - Final eval: 115 examples (reserved in step 4, held-out from all training)

### 3. Generate Preference Data

- **Status**: Implementation complete, awaiting execution
- **Prerequisites**: Trained SFT model, prompt pools from step 2
- **Script**: `python src/sft/generate_preferences.py`
- **Output**: 2 responses per prompt for both pools
- **Estimated Time**: 2-4 hours

### 4. Verify Outputs

- **Status**: Implementation complete, awaiting execution
- **Prerequisites**: All outputs from previous steps
- **Script**: `python src/eval/evaluate_model.py`
- **Output**: Evaluation samples and quality checks

## 📋 Instructions for Execution

See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for detailed step-by-step instructions.

## 🚀 Future Work (Beyond SFT)

These are the next phases of the project after SFT is complete:

- [ ] RLAIF Stage 1: AI grader preference labeling
- [ ] RLAIF Stage 2: DPO training on AI preferences
- [ ] RLHF: Human preference labeling (300 prompts)
- [ ] RLHF-only: DPO training on human preferences
- [ ] RLAIF→RLHF: Two-stage DPO (AI then human)
- [ ] Final evaluation: Compare all 4 models
- [ ] Grader model setup for ranking
- [ ] Analysis and write-up for CS230 report
