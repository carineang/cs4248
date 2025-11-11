# CS4248 Machine Translation Project

This repository contains scripts for Chinese-English machine translation using mT5 models. The pipeline includes dataset preparation, model training, inference, and evaluation.

## Quick Start

### 1. Environment Setup

First, set up the conda environment and install dependencies:

```bash
# Setup conda environment (creates 'mt_env')
bash env-setup-miniconda.sh
# Or submit via SLURM
sbatch env-setup-miniconda.sh

# Activate environment
conda activate mt_env

# Install additional dependencies if needed
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### Download WMT Dataset
```bash
mtdata get-recipe -ri wmt22-zhen -o wmt22-zhen
```

#### Tokenize Dataset
```bash
# Edit paths in tokenizer/tokenizer_streaming.sh if needed
bash tokenizer/tokenizer_streaming.sh
# Or submit via SLURM
sbatch tokenizer/tokenizer_streaming.sh
```

#### Merge Tokenized Chunks
```bash
# Edit paths in tokenizer/chunk_merging.sh if needed
bash tokenizer/chunk_merging.sh
# Or submit via SLURM
sbatch tokenizer/chunk_merging.sh
```

#### (Optional) Create Dataset Subset
```bash
# Create a 5% subset
bash create_subset.sh
# Or manually
python create_dataset_subset.py --input ./tokenized_dataset/WMT22_Train_Merged --output ./tokenized_dataset/WMT22_Train_Merged_5pct --percentage 0.05
```

#### (Optional) Verify Subset
```bash
bash verify_subset.sh
# Or manually
python verify_subset.py --original ./tokenized_dataset/WMT22_Train_Merged --subset ./tokenized_dataset/WMT22_Train_Merged_5pct --samples 3
```

## Model Training

### Configure Training

1. Copy and modify a configuration file from `configs/`:
   ```bash
   cp configs/mT5-base-training-multi-gpu.yaml configs/training.yaml
   # Edit training.yaml with your settings
   ```

2. Available pre-configured options:
   - `mT5-small-training-multi-gpu.yaml` - Small model, multi-GPU
   - `mT5-base-training-multi-gpu.yaml` - Base model, multi-GPU  
   - `mT5-large-training-multi-gpu.yaml` - Large model, multi-GPU
   - `mT5-*-training-lora-multi-gpu.yaml` - LoRA fine-tuning versions
   - `mT5-large-training-single-gpu.yaml` - Large model, single GPU

### Run Training

#### Single GPU Training
```bash
bash train_single_gpu.sh
# Or submit via SLURM
sbatch train_single_gpu.sh
```

#### Multi-GPU Training
```bash
bash train_multi_gpu.sh
# Or submit via SLURM
sbatch train_multi_gpu.sh
```

#### Manual Training
```bash
python train_mt.py --config ./configs/training.yaml
```

## Inference

### Single Text Translation
```bash
python inference.py \
    --model-path ./models/mt5-base-finetuned/checkpoint-XXX \
    --input-text "你好世界"
```

### Batch File Translation
```bash
python inference.py \
    --model-path ./models/mt5-base-finetuned/checkpoint-XXX \
    --input-file ./dataset/test_chinese.txt \
    --output-file ./outputs/translations.txt \
    --batch-size 32
```

### SLURM Inference
```bash
# Edit paths in inference.sh
sbatch inference.sh
```

## Evaluation

### Run Evaluation Script
```bash
# Edit paths in evaluation.sh for your model and test files
sbatch evaluation.sh
```

### Manual Evaluation
The evaluation script typically calculates BLEU scores and other metrics using:
- BLEU score via `sacrebleu`
- COMET scores via `unbabel-comet`

## Configuration

### Training Configuration Example
```yaml
training_dataset_paths: 
  - "./tokenized_dataset/ALMA_Human_Parallel"
evaluation_dataset_paths:
  - "./tokenized_dataset/Tatoeba"
model_name: "google/mt5-base"
use_lora: False  # Set to True for LoRA fine-tuning
training_args:
  output_dir: "./models/mt5-base-finetuned"
  eval_strategy: "epoch"
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 1
  num_train_epochs: 10
  # Add more training arguments as needed
```

## Project Structure

```
├── README.md
├── requirements.txt
├── env-setup-miniconda.sh      # Environment setup
├── train_mt.py                 # Main training script
├── train_single_gpu.sh         # Single GPU training
├── train_multi_gpu.sh          # Multi-GPU training
├── inference.py                # Inference script
├── inference.sh                # SLURM inference
├── evaluation.sh               # Evaluation script
├── configs/                    # Training configurations
│   ├── training.yaml          # Main config (copy from others)
│   ├── mT5-base-training-*.yaml
│   └── mT5-large-training-*.yaml
├── dataset/                    # Raw datasets
├── tokenized_dataset/          # Processed datasets
├── tokenizer/                  # Tokenization scripts
│   ├── tokenizer_streaming.py
│   ├── tokenizer_streaming.sh
│   ├── chunk_merging.py
│   └── chunk_merging.sh
├── create_dataset_subset.py    # Dataset subset creation
├── create_subset.sh
├── verify_subset.py           # Subset verification
├── verify_subset.sh
├── models/                    # Trained models (created during training)
├── logs/                      # Training/evaluation logs
└── outputs/                   # Inference outputs
```