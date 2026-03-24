# Developer Guide

This document covers setup, dataset preparation, running experiments, and 
visualizing results. See [README.md](./README.md) for a project overview.

## Setup and Installation

### Prerequisites

- **OS:** Linux (tested on Ubuntu 22.04)
- **GPU:** NVIDIA GPU with CUDA 12.4 compatible drivers
- **Conda:** Anaconda or Miniconda
- **Poetry:** Python dependency manager

To install Poetry and configure it to work with Conda:
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false
```

### Installation

1. **Create and activate a Conda environment:**
```bash
    conda create -n mdetector_env python=3.9 -y
    conda activate mdetector_env
```

2. **Install dependencies with Poetry:**
```bash
    poetry install
```

3. **Always prefix commands with `poetry run`:**
```bash
    poetry run python scripts/generate_labels.py --help
```

### Using a Different CUDA Version

`pyproject.toml` is pre-configured for CUDA 12.4. To use a different version 
(e.g., 11.8):

1. Update the PyTorch source URL in `pyproject.toml`:
```toml
    [[tool.poetry.source]]
    name = "pytorch"
    url = "https://download.pytorch.org/whl/cu118"
    priority = "explicit"
```

2. Update the `torch-scatter` URL:
    - Go to the [PyG wheel page](https://data.pyg.org/whl/)
    - Find the section for your PyTorch + CUDA version
    - Find the `.whl` matching your Python version (`cp39`) and platform 
      (`linux`)
    - Replace the existing `torch-scatter` URL in `pyproject.toml`

3. Re-lock and install:
```bash
    poetry lock --no-update
    poetry install
```

## Dataset and Ground Truth Preparation

The algorithm requires the [nuScenes dataset](https://www.nuscenes.org/). 
Set the `dataroot` path in `config/m_detector_config.yaml` to point to your 
local copy. You can also configure which dataset version to use 
(e.g., `v1.0-mini` or `v1.0-trainval`).

### Generate Ground Truth Labels

The script `scripts/generate_labels.py` generates ground truth files 
identifying dynamic points in each sweep. It interpolates 3D bounding boxes 
for every sweep (not just keyframes), labels points inside boxes above a 
velocity threshold, and saves the result as compressed `.pt` files.
```bash
# Specific scenes
poetry run python scripts/generate_labels.py \
    --config config/m_detector_config.yaml \
    --scenes 0,1,5

# All scenes
poetry run python scripts/generate_labels.py \
    --config config/m_detector_config.yaml \
    --scenes all
```

### Verify Ground Truth Visually
```bash
poetry run python scripts/visualize_gt.py \
    --config config/m_detector_config.yaml \
    --scene-index 0 \
    --output-path scene_0_gt.mp4
```

Output colors: blue = ground truth dynamic, grey = static, with 3D bounding 
boxes drawn for all dynamic objects.

## Running Experiments

The tuning pipeline has three stages.

### Stage 1: Tune Geometric Parameters

Tunes core detection parameters (occlusion thresholds, MCC settings). 
Refinement is disabled for speed.
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

poetry run python scripts/run_experiment.py \
    --mode tune-full \
    --study-name geo_study_v1 \
    --n-trials 500
```

### Stage 2: Bake Intermediate Results

Takes the best parameters from Stage 1, runs the geometric pipeline, and 
caches the pre-refinement results to avoid re-running the expensive pipeline 
during Stage 3.
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

poetry run python scripts/run_experiment.py \
    --mode bake \
    --study-name "Baking geo_study_v1" \
    --source-study-name geo_study_v1
```

### Stage 3: Tune Refinement Parameters

Loads baked data and tunes only the HDBSCAN clustering and convex hull 
parameters. Much faster than tuning everything at once.

The `--bake-id` is printed at the end of the bake command.
```bash
poetry run python scripts/run_experiment.py \
    --mode tune-refinement \
    --study-name refinement_on_geo_v1 \
    --n-trials 1000 \
    --bake-id geo_study_v1_trial388
```

## Visualizing Results

### Step 1: Export Best Parameters
```bash
poetry run python scripts/export_best_params.py \
    --storage "sqlite:///path/to/your/database.db" \
    --study-name "geo_study_v1" \
    --output-file "best_params_from_study.yaml"
```

### Step 2: Process a Scene
```bash
poetry run python scripts/process_scene.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --params "best_params_from_study.yaml" \
    --output-path "results/scene-1.pt"
```

### Step 3a: Generate a Video
```bash
poetry run python scripts/visualize_mdetector.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --processed-file "results/scene-1.pt" \
    --output-path "output.mp4"
```

### Step 3b: Interactive Error Analyzer
```bash
poetry run python scripts/analyze_errors_interactive.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --processed-file "results/scene-1.pt"
```

**Controls:** `Space` play/pause · `.` next frame · `,` previous frame · 
`Q` quit

**Point colors:**
- Green — True Positive (moving, correctly labeled dynamic)
- Red — False Positive (static, incorrectly labeled dynamic)
- Orange — False Negative (moving, missed by detector)
- Grey — True Negative (static, correctly labeled static)
- Blue boxes — Ground truth bounding boxes