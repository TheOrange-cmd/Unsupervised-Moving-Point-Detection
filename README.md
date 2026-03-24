# Unsupervised Moving Point Detection in LiDAR

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A GPU-accelerated PyTorch re-implementation of **M-Detector** [1], an 
unsupervised algorithm for detecting moving points in LiDAR point cloud 
sequences. Developed as a personal research project exploring dynamic object 
detection for autonomous driving, evaluated on the 
[nuScenes dataset](https://www.nuscenes.org/).

## Demo

<video src="demo.mp4" controls width="100%"></video>

*A nuScenes scene with points labeled as dynamic (red), ground (green), 
or static background (grey).*

## What This Does

The core idea of M-Detector is to identify moving points **without any 
supervision or learned priors**, purely by detecting geometric inconsistencies 
over time: a point is considered dynamic if it consistently occludes parts of 
an established static map of the environment.

The pipeline consists of five stages:

- **Ground removal** — GPU-accelerated RANSAC to pre-label ground points
- **Occlusion pass** — fast coarse check against recent history to generate 
  initial dynamic candidates
- **Map Consistency Check (MCC)** — filters candidates against a longer-term 
  static map to reduce false positives
- **Event-based tests** — handles edge cases such as objects moving toward or 
  away from the sensor
- **Frame refinement** — HDBSCAN clustering with convex hull expansion for 
  cleaner object-level detections

## Results

Evaluated on nuScenes, the geometric pipeline achieves reasonable dynamic 
point recall. The frame refinement stage showed limited improvement over the 
geometric baseline — bidirectional processing is a natural next step that was 
outside the scope of this project. Overall performance does not match 
state-of-the-art supervised methods, which is expected for a purely 
unsupervised approach on a challenging real-world dataset.

The primary contribution is the re-implementation itself and the surrounding 
evaluation infrastructure, which provides a clean, reproducible baseline for 
further experimentation.

## Key Features

- **Fully GPU-accelerated** — vectorized PyTorch operations throughout
- **Three-stage tuning pipeline** — `tune-full` → `bake` → `tune-refinement` 
  using Optuna and Ray for efficient parallel hyperparameter search
- **Interactive error analysis** — Open3D-based visualization tool for 
  frame-by-frame inspection of true/false positives and negatives
- **Configurable** — all parameters managed via a central `config.yaml`

## Project Structure
```
.
├── config/
│   └── m_detector_config.yaml  # Main configuration file
├── src/
│   ├── core/                   # Core algorithm (occlusion, MCC, event tests)
│   ├── data_utils/             # Ground truth generation and validation
│   ├── tuning/                 # Ray + Optuna experiment infrastructure
│   └── utils/
└── scripts/                    # Entry points for all pipeline stages
```

For setup instructions, dataset preparation, and full usage documentation, 
see [DEVELOPER.md](./DEVELOPER.md).

## References

[1] Wu, H., Li, Y., Xu, W. et al. Moving event detection from LiDAR point 
streams. *Nature Communications* 15, 345 (2024). 
https://doi.org/10.1038/s41467-023-44554-8