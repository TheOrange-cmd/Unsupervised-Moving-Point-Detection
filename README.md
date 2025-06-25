# Unsupervised Moving Point Detection in LiDAR

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a PyTorch-based re-implementation of M-Detector[1], an algorithm for detecting moving points within LiDAR sequences. The core principle of the detector is to identify dynamic points by analyzing geometric inconsistencies over time against a dynamically built map of the static environment.

## Overview

The M-Detector algorithm processes a sequence of LiDAR sweeps to build a temporary map of the world. For each new sweep, it projects the points into historical frames and checks for occlusions. Points that consistently occlude parts of the established static map are labeled as dynamic. The system includes several stages of filtering and refinement to improve accuracy:

- **Initial Occlusion Pass:** A fast, coarse check against recent history to generate initial candidates.
- **Map Consistency Check (MCC):** A crucial filter that verifies dynamic candidates against a longer-term static map to reduce false positives.
- **Event-Based Tests:** Logic to handle specific scenarios, such as objects moving towards or away from the sensor.
- **Frame Refinement:** A final clustering stage (using HDBSCAN) to clean up noise and fill in object detections using convex hulls. Note that the frame refinement is currently not showing promising results. 

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

### Installation Steps

1.  **Create and activate the Conda environment:**
    The environment is based on Python 3.9.

    ```bash
    conda create -n mdetector_env python=3.9 -y
    conda activate mdetector_env
    ```

2.  **Install core dependencies with Conda:**
    This includes libraries for data science, parallelization, and clustering.

    ```bash
    conda install -c conda-forge numpy pandas matplotlib pyyaml scipy scikit-learn ray-all optuna hdbscan -y
    ```

3.  **Install the NuScenes Devkit:**

    ```bash
    python -m pip install nuscenes-devkit
    ```

4.  **Install PyTorch for CUDA 12.4:**
    It is crucial to install the correct PyTorch build for your CUDA version. The `--index-url` points to the official PyTorch wheel index.

    ```bash
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```

5.  **Install `torch-scatter`:**
    This library is a required dependency for some advanced PyTorch operations. It must be installed from a separate wheel index that matches your specific PyTorch and CUDA versions.

    ```bash
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0%2Bcu124.html
    ```

### System Compatibility Notes

-   **Windows/macOS:** This project has only been tested on Linux. Running on Windows may lead to issues with file paths or the `ray-all` installation. macOS does not support NVIDIA CUDA, so the code would have to run on the CPU, which would be prohibitively slow.
-   **Different CUDA/PyTorch Versions:** If you have a different CUDA version (e.g., 11.8), you **must** find the corresponding wheel URLs for PyTorch (from the [PyTorch website](https://pytorch.org/get-started/previous-versions/)) and `torch-scatter` (from the [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).

## Dataset Preparation

The algorithm is designed to work with the [NuScenes dataset](https://www.nuscenes.org/nuscenes). You will need to download the dataset and place it in a location accessible by the scripts. You can select which version of the dataset (e.g., 'v1.0-mini' or 'v1.0-trainval') to use in the config file. 

### Ground Truth Label Generation

For tuning and validation, the project requires ground truth labels that identify which points in the raw point cloud are dynamic.

> **TODO:**
>
> - [ ] Convert the existing Jupyter Notebook for label generation into a user-friendly Python script (`src/data_utils/generate_labels.py`).
> - [ ] Add clear instructions to this README on how to run the script to generate the required `.pt` ground truth files. The script should take the config file and an output directory as arguments.

## Running Experiments

The hyperparameter tuning process is split into three main stages. The scripts are located in `src/ray_scripts/`.

### 1. Tune Geometric Parameters (`tune-full`)

This stage tunes the core parameters of the detection algorithm, such as occlusion thresholds and map consistency settings. The refinement stage is disabled during this phase for speed.

```bash
# Set this environment variable to prevent CUDA out-of-memory errors on some systems
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the full geometric tuning
python -m src.ray_scripts.run_experiment \
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
python -m src.ray_scripts.run_experiment \
    --mode bake \
    --study-name "Baking geo_study_v1" \
    --source-study-name geo_study_v1
```

### 3. Tune Refinement Parameters (tune-refinement)

This final stage loads the baked data and tunes only the frame refinement parameters (e.g., HDBSCAN clustering settings) to find the best post-processing configuration. This is much faster than tuning all parameters at once.

Note: The --bake-id will be printed at the end of the bake command. Use that ID here.

```bash
python -m src.ray_scripts.run_experiment \
    --mode tune-refinement \
    --study-name refinement_on_geo_v1 \
    --n-trials 1000 \
    --bake-id geo_study_v1_trial388
```

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
│   ├── ray_scripts/            # Scripts for running experiments
│   │   ├── modes/              # Logic for bake, tune-full, tune-refinement
│   │   ├── run_experiment.py   # Main entry point
│   │   └── ...
│   └── utils/                  # General utility functions
└── ...
```


### References

[1] Wu, H., Li, Y., Xu, W. et al. Moving event detection from LiDAR point streams. Nat Commun 15, 345 (2024). https://doi.org/10.1038/s41467-023-44554-8