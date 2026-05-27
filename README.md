# Backdoor Attack Pipeline for Person-in-WiFi-3D (3D Pose Regression)

This repository contains a complete, reproducible scientific experimental pipeline to evaluate backdoor vulnerability on the CVPR 2024 **Person-in-WiFi-3D** dataset under three standard baseline attacks: **FTrojan**, **SIG**, and **WaNet**, adapted for continuous 3D coordinate pose regression.

---

## 1. Project Directory Structure
The workspace is organized as follows:
```
project_root/
├── configs/                # Automatically generated YAML configuration files
├── data/
│   ├── raw/                # Place raw dataset files here
│   ├── processed/          # Preprocessed data tensors (.npy)
│   └── splits/             # Train/Val/Test division splits (.json)
├── src/
│   ├── dataset/            # PersonInWiFi3DDataset and transforms
│   ├── attacks/            # Backdoor adapters (FTrojan, SIG, WaNet) and target poses
│   ├── models/             # Lightweight CNN + Transformer model with gradient checkpointing
│   ├── trainers/           # Trainer with L1 loss, AMP, and dynamic batch scaling
│   ├── evaluation/         # MPJPE, PCK@10/20/30, and ASR metric calculations
│   ├── defenses/           # STRIP adaptation using output variance
│   └── visualization/      # High-quality visualization plotting utilities
├── outputs/
│   ├── figures/            # Output plots (ASR vs. Poison Rate, 3D Poses,heatmaps)
│   ├── tables/             # Summary tables in CSV, Markdown, and LaTeX
│   ├── logs/               # Output execution logs
│   ├── strip/              # Raw defense scores
│   └── reports/            # Markdown summary and experimental reports
├── checkpoints/            # Model checkpoints per attack baseline
├── tests/                  # Unit and integration test suite
├── requirements.txt        # Virtual environment dependencies
├── run_pipeline.py         # Master pipeline script
└── README.md               # User guide
```

---

## 2. Methodology & Regression Backdoor Formulation
Unlike typical image classification where backdoors map triggered inputs to a target class index, this project implements a **pose regression backdoor formulation**:
- **Target Pose**: Attacker-specified target pose (e.g. T-pose or neutral standing pose).
- **Attack Success Rate (ASR)**: The percentage of triggered samples where the Mean Per Joint Position Error (MPJPE) between the model's output prediction and the attacker's target pose is below a specified threshold (default `80mm`).
- **STRIP Defense for Regression**: Measures output joint coordinate variance under input blending. Since the trigger forces the model to output a fixed target pose, poisoned inputs show extremely low output variance across random blends compared to clean inputs.

---

## 3. Getting Started

### Prerequisite Dependencies
Install the required Python packages into your environment:
```bash
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Running the End-to-End Pipeline
You can run the entire experiment (folder setup, dataset generation/verification, clean and attacked model training, evaluation, STRIP analysis, table exports, and figure plotting) with a single command:
```bash
# Run a quick validation demo (3 epochs per model, fast mock dataset)
uv run python run_pipeline.py --quick_run True

# Run full rigorous training on the full raw Person-in-WiFi-3D train/val/test splits
uv run python run_pipeline.py --quick_run False --epochs 20

# Run on two idle CUDA GPUs selected from nvidia-smi
python run_pipeline.py --quick_run False --epochs 20 --gpus auto --num_gpus 2

# Pin specific GPUs when the server owner assigns them
python run_pipeline.py --quick_run False --epochs 20 --gpus 0,1 --batch_size 64
```

### Running the Test Suite
Verify that all modules, shape flows, triggers, models, and metrics are functioning correctly:
```bash
uv run python -m unittest tests/test_pipeline.py
```

---

## 4. Output Artifacts
All generated plots are saved under `outputs/figures/`, tables under `outputs/tables/`, and reports under `outputs/reports/`. The final CSV/Markdown summary provides clean comparisons of Clean, FTrojan, SIG, and WaNet performance.
