# Unsupervised Moving Point Detection in LiDAR

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a PyTorch-based re-implementation of M-Detector[1], an algorithm for detecting moving points within LiDAR sequences. The core principle of the detector is to identify dynamic points by analyzing geometric inconsistencies over time against a dynamically built map of the static environment.

## Overview

The M-Detector algorithm processes a sequence of LiDAR sweeps to build a temporary map of the world. For each new sweep, it projects the points into historical frames and checks for occlusions. Points that consistently occlude parts of the established static map are labeled as dynamic. The system includes several stages of filtering and refinement to improve accuracy:

- **Initial Ground Removal:** A simple and fast torch-based RANSAC implementation to pre-label ground points as static. This will later be extended to more robust ground removal using TerraSeg. 
- **Occlusion Pass:** A fast, coarse check against recent history to generate initial candidates.
- **Map Consistency Check (MCC):** A crucial filter that verifies dynamic candidates against a longer-term static map to reduce false positives.
- **Event-Based Tests:** Logic to handle specific scenarios, such as objects moving towards or away from the sensor, using more detailed occlusion checks.
- **Frame Refinement:** A final clustering stage (using HDBSCAN) to clean up noise and fill in object detections using convex hulls. Note that the frame refinement is currently not showing promising results, but is left for future work, e.g. expansion to bidirectional processing. 

The entire experimentation pipeline is managed using **Ray** for parallel processing and **Optuna** for multi-stage hyperparameter tuning.

## Key Features

- **PyTorch & CUDA:** Fully vectorized and GPU-accelerated for high performance.
- **Tuning Pipeline:** Implements a three-stage parallel tuning process (`tune-full` -> `bake` -> `tune-refinement`) using Optuna and Ray.
- **Configurable:** All algorithm parameters are managed via a central `config.yaml` file.

## Setup and Installation

### Prerequisites

- **OS:** Linux (tested on Ubuntu 22.04).
- **GPU:** An NVIDIA GPU with CUDA 12.4 compatible drivers.
- **Conda:** Anaconda or Miniconda package manager.
- **Poetry:** The Python dependency manager. Install it with `pip install poetry`.

### Installation Steps

This project uses **Poetry** to manage dependencies for a reproducible environment.

1.  **Create and activate a Conda environment:**
    This environment will provide the base Python 3.9 interpreter.

    ```bash
    conda create -n mdetector_env python=3.9 -y
    conda activate mdetector_env
    ```

2.  **Install project dependencies with Poetry:**
    This single command reads the `pyproject.toml` file, resolves all dependencies (including PyTorch, devkit, etc.), and installs them.

    ```bash
    poetry install
    ```

3.  **Run scripts using `poetry run`:**
    To ensure you are using the correct environment, always prefix your commands with `poetry run`. For example:
    ```bash
    poetry run python scripts/generate_labels.py --help
    ```

### Handling Different CUDA Versions

The `pyproject.toml` file is pre-configured for **CUDA 12.4**. If you are using a different CUDA version (e.g., 11.8), you **must** manually edit `pyproject.toml` before running `poetry install`.

1.  **Update the PyTorch source URL:** Change `cu124` to your version (e.g., `cu118`).
    ```toml
    [[tool.poetry.source]]
    name = "pytorch"
    url = "https://download.pytorch.org/whl/cu118" # Changed from cu124
    priority = "explicit"
    ```

2.  **Update the `torch-scatter` URL:**
    - Go to the [PyG wheel page](https://data.pyg.org/whl/).
    - Find the section for your PyTorch version (e.g., `torch-2.1.0+cu118`).
    - Find the wheel (`.whl`) file that matches your Python version (`cp39`) and platform (`linux`).
    - Copy the full URL and replace the existing `torch-scatter` URL in `pyproject.toml`.

After editing the file, run `poetry lock --no-update` followed by `poetry install`.

## Dataset and Ground Truth Preparation

The algorithm is designed to work with the [NuScenes dataset][URL TRUNCATED]. You will need to download the dataset and ensure the `dataroot` path in `config/m_detector_config.yaml` points to its location. You can also select which version of the dataset (e.g., `v1.0-mini` or `v1.0-trainval`) to use in the config.

### 1. Ground Truth Label Generation

For tuning and validation, the project requires ground truth files that identify which points in each sweep are dynamic. The script `scripts/generate_labels.py` handles this.

**What it does:**
- It processes each scene sweep by sweep.
- It interpolates/extrapolates the ground truth 3D bounding boxes for every single sweep (not just keyframes).
- It identifies all LiDAR points that fall within a box whose velocity is above a configured threshold.
- **Note:** The same point pre-filtering logic defined in the config (`point_pre_filtering`) is applied during label generation to ensure consistency with the main detector.
- It saves the indices of these dynamic points into a compressed `.pt` file for each scene.

**How to run it:**
Provide the path to your config file and specify which scenes to process.

```bash
# Generate labels for specific scenes (e.g., 0, 1, and 5)
poetry run python scripts/generate_labels.py \
    --config config/m_detector_config.yaml \
    --scenes 0,1,5

# Or generate labels for all scenes in the dataset
poetry run python scripts/generate_labels.py \
    --config config/m_detector_config.yaml \
    --scenes all
```

## 2. Visual Verification of Ground Truth

To ensure the generated labels are correct, you can use scripts/visualize_gt.py to create a video of the results.

**What it does:**

-    Renders a video for a specified scene.
-    Colors the point cloud: blue points are ground truth dynamic, grey points are static.
-    Draws the 3D bounding boxes for all dynamic objects in the frame.

How to run it:

```bash
# Create a video for scene 0 and save it as scene_0061_gt.mp4
poetry run python scripts/visualize_gt.py \
    --config config/m_detector_config.yaml \
    --scene-index 0 \
    --output-path scene_0061_gt.mp4
```

## Running Experiments

The tuning process is split into three main stages. The scripts are located in `scripts`.

### 1. Tune Geometric Parameters (`tune-full`)

This stage tunes the core parameters of the detection algorithm, such as occlusion thresholds and map consistency settings. The refinement stage is disabled during this phase for speed.

```bash
# Set this environment variable to prevent CUDA out-of-memory errors on some systems
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the full geometric tuning
python -m scripts.run_experiment \
    --mode tune-full \
    --study-name geo_study_v1 \
    --n-trials 500
```

### 2. Bake Intermediate Results (bake)

This stage takes the best parameters from the geo_study_v1 study, runs the geometric pipeline on the dataset, and saves ("bakes") the intermediate results (labels before refinement) to a cache file. This avoids re-running the expensive geometric pipeline when tuning the refinement stage.

```bash
# Set this environment variable to prevent CUDA out-of-memory errors on some systems
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the full geometric tuning
python -m scripts.run_experiment \
    --mode bake \
    --study-name "Baking geo_study_v1" \
    --source-study-name geo_study_v1
```

### 3. Tune Refinement Parameters (tune-refinement)

This final stage loads the baked data and tunes only the frame refinement parameters (e.g., HDBSCAN clustering settings) to find the best post-processing configuration. This is much faster than tuning all parameters at once.

Note: The --bake-id will be printed at the end of the bake command. Use that ID here.

```bash
python -m scripts.run_experiment \
    --mode tune-refinement \
    --study-name refinement_on_geo_v1 \
    --n-trials 1000 \
    --bake-id geo_study_v1_trial388
```

## Visualizing and Analyzing Results

After completing the tuning process, the next step is to visually inspect the performance of the best model. This workflow allows you to take the best hyperparameters from an Optuna study, run the detector on a specific scene, and then either generate a video of the output or launch an interactive 3D analysis tool.

The scripts for this process are located in the `scripts/` directory.

### Step 1: Export Best Parameters

First, extract the best-performing hyperparameters from your Optuna study database. This script connects to the database, automatically finds the trial with the best score in a given study, and saves its parameters to a new YAML file.

```bash
poetry run python scripts/export_best_params.py \
    --storage "sqlite:///path/to/your/database.db" \
    --study-name "your_study_name" \
    --output-file "best_params_from_study.yaml"
```

*   `--storage`: The connection string for your Optuna database.
*   `--study-name`: The name of the study you want to pull the best trial from.
*   `--output-file`: The name of the YAML file where the best parameters will be saved.

### Step 2: Process a Scene (Bake Results)

Next, run the M-Detector on a specific scene using the parameters you just exported. This script "bakes" the results by saving the final output for each frame—including point coordinates, labels, and the original point indices needed for error analysis—to a single `.pt` file. This avoids re-running the detector every time you want to view the results.

Make sure the output directory (e.g., `results/`) exists before running.

```bash
# Example for scene index 1
poetry run python scripts/process_scene.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --params "best_params_from_study.yaml" \
    --output-path "results/scene-1_best_params_with_indices.pt"
```

*   `--scene-index`: The index of the nuScenes scene you want to process.
*   `--params`: The YAML file containing the best parameters from Step 1.
*   `--output-path`: The file where the processed results will be stored.

### Step 3: Generate Visualizations

With the processed results file, you can now generate visualizations without needing to run the detector again.

#### Option A: Generate a Video

This script reads the processed file and renders a video of the detector's output, showing dynamic points in red and ground points in green.

```bash
poetry run python scripts/visualize_mdetector.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --processed-file "results/scene-1_best_params_with_indices.pt" \
    --output-path "video_scene-1_best_params.mp4"
```

*   `--processed-file`: The input `.pt` file generated in Step 2.
*   `--output-path`: The path where the final `.mp4` video will be saved.

#### Option B: Launch the Interactive Analyzer

This script launches an interactive Open3D window for in-depth error analysis. It provides full camera control and playback functionality.

```bash
poetry run python scripts/analyze_errors_interactive.py \
    --config "config/m_detector_config.yaml" \
    --scene-index 1 \
    --processed-file "results/scene-1_best_params_with_indices.pt"
```
**Interactive Controls:**
*   **Mouse:** Pan, zoom, and rotate the camera.
*   **`[Spacebar]`:** Play or pause the animation.
*   **`[.]` (Period):** Advance one frame forward (when paused).
*   **`[,]` (Comma):** Go back one frame (when paused).
*   **`[Q]`:** Quit the application.

**Point Cloud Colors:**
*   <span style="color:green">■</span> **Green (True Positive):** A moving point correctly labeled as dynamic.
*   <span style="color:red">■</span> **Red (False Positive):** A static point incorrectly labeled as dynamic.
*   <span style="color:orange">■</span> **Orange (False Negative):** A moving point that the detector missed.
*   <span style="color:gray">■</span> **Grey (True Negative):** A static point correctly labeled as static.
*   <span style="color:blue">▬</span> **Blue Boxes:** Ground truth bounding boxes for moving objects.

## Project Structure

```
.
├── config/
│   └── m_detector_config.yaml  # Main configuration file
├── src/
│   ├── core/                   # Core algorithm logic
│   │   ├── m_detector/         # M-Detector modules (occlusion, mcc, etc.)
│   │   ├── depth_image.py      # Main DepthImage data structure
│   │   └── ...
│   ├── data_utils/             # Utils for data processing and validation
│   │   ├── label_generation.py # GT label generation logic
│   │   └── ...
│   ├── tuning/                 # Scripts for running experiments
│   │   ├── modes/              # Logic for bake, tune-full, tune-refinement
│   │   ├── ray_actors.py       # Main file for ray utilities
│   │   └── ...
│   └── utils/                  # General utility functions
└── ...
```


### References

[1] Wu, H., Li, Y., Xu, W. et al. Moving event detection from LiDAR point streams. Nat Commun 15, 345 (2024). https://doi.org/10.1038/s41467-023-44554-8