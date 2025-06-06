# src/core/m_detector/tests/benchmark_occlusion_checks.py

import time
import numpy as np
import torch
import os
import yaml
import sys

from src.config_loader import MDetectorConfigAccessor
from src.core.m_detector.base import MDetector
from src.core.depth_image import DepthImage as DepthImageTorch
from src.core.depth_image_legacy import DepthImage as DepthImageLegacy

def generate_realistic_point_cloud(num_points=35000, max_range=80.0):
    """Generates a random point cloud that looks roughly like a LiDAR scan."""
    # Generate points in a spherical distribution
    radius = np.random.uniform(1.0, max_range, num_points)
    # phi is horizontal angle, theta is vertical
    phi = np.random.uniform(-np.pi, np.pi, num_points)
    theta = np.random.uniform(np.deg2rad(-15), np.deg2rad(15), num_points)
    
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta)
    
    return np.stack([x, y, z], axis=-1).astype(np.float32)

def setup_benchmark_environment():
    """Sets up a detector instance and historical DIs for testing."""
    # --- Create a dummy config ---
    config_path = "temp_benchmark_config.yaml"
    dummy_config = {
        'm_detector': {
            'depth_image': {
                'resolution_h_deg': 0.4, 'resolution_v_deg': 1.6,
                'phi_min_rad': -3.14159, 'phi_max_rad': 3.14159,
                'theta_min_rad': -0.2618, 'theta_max_rad': 0.2618,
            },
            'occlusion_determination': {
                'epsilon_depth': 0.3,
                'pixel_neighborhood_h': 1,
                'pixel_neighborhood_v': 1,
            }
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(dummy_config, f)

    config_accessor = MDetectorConfigAccessor(config_path)
    detector = MDetector(config_accessor)
    
    # --- Create historical Depth Images ---
    print("Setting up benchmark environment...")
    print("  Generating historical point cloud...")
    hist_points = generate_realistic_point_cloud(num_points=35000)
    hist_pose = np.eye(4, dtype=np.float32)
    hist_timestamp = 1000.0

    print("  Populating NumPy-based historical DepthImage...")
    start_time = time.time()
    hist_di_numpy = DepthImageLegacy(hist_pose, detector.config_accessor.get_depth_image_params(), hist_timestamp)
    hist_di_numpy.add_points_batch(hist_points)
    print(f"    Done in {time.time() - start_time:.4f} seconds.")

    print("  Populating PyTorch-based historical DepthImage (CPU)...")
    device_cpu = torch.device("cpu")
    start_time = time.time()
    hist_di_torch_cpu = DepthImageTorch(hist_pose, detector.config_accessor.get_depth_image_params(), hist_timestamp, device_cpu)
    hist_di_torch_cpu.add_points_batch(hist_points)
    print(f"    Done in {time.time() - start_time:.4f} seconds.")
    
    hist_di_torch_gpu = None
    if torch.cuda.is_available():
        print("  Populating PyTorch-based historical DepthImage (GPU)...")
        device_gpu = torch.device("cuda")
        start_time = time.time()
        hist_di_torch_gpu = DepthImageTorch(hist_pose, detector.config_accessor.get_depth_image_params(), hist_timestamp, device_gpu)
        hist_di_torch_gpu.add_points_batch(hist_points)
        torch.cuda.synchronize() # Wait for GPU population to finish
        print(f"    Done in {time.time() - start_time:.4f} seconds.")
    else:
        print("  CUDA not available, skipping GPU setup.")

    # Clean up dummy config file
    os.remove(config_path)

    return detector, hist_di_numpy, hist_di_torch_cpu, hist_di_torch_gpu


def run_benchmark(detector, hist_di_numpy, hist_di_torch_cpu, hist_di_torch_gpu):
    """Runs the timing comparison for check_occlusion_batch."""
    num_points = 35000
    num_runs = 10  # Average over 10 runs for stability

    print("\n--- Starting Benchmark: check_occlusion_batch ---")
    print(f"  Number of points per call: {num_points}")
    print(f"  Number of runs per version: {num_runs}")

    # --- Generate a consistent "current" point cloud for all tests ---
    current_points_np = generate_realistic_point_cloud(num_points)

    # --- 1. Legacy NumPy Benchmark ---
    print("\n[1] Benchmarking Legacy NumPy version (CPU)...")
    detector.device = torch.device("cpu") # Ensure detector is in CPU mode for this
    legacy_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = detector.check_occlusion_batch_legacy(current_points_np, hist_di_numpy)
        end_time = time.time()
        legacy_times.append(end_time - start_time)
    avg_legacy_time = np.mean(legacy_times)
    print(f"  Average time: {avg_legacy_time:.6f} seconds")

    # --- 2. PyTorch CPU Benchmark ---
    print("\n[2] Benchmarking PyTorch version (CPU)...")
    detector.device = torch.device("cpu")
    current_points_torch_cpu = torch.from_numpy(current_points_np).to(detector.device)
    torch_cpu_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = detector.check_occlusion_batch(current_points_torch_cpu, hist_di_torch_cpu)
        end_time = time.time()
        torch_cpu_times.append(end_time - start_time)
    avg_torch_cpu_time = np.mean(torch_cpu_times)
    print(f"  Average time: {avg_torch_cpu_time:.6f} seconds")

    # --- 3. PyTorch GPU Benchmark ---
    if hist_di_torch_gpu is not None and torch.cuda.is_available():
        print("\n[3] Benchmarking PyTorch version (GPU)...")
        detector.device = torch.device("cuda")
        current_points_torch_gpu = torch.from_numpy(current_points_np).to(detector.device)
        torch_gpu_times = []
        for _ in range(num_runs):
            # Synchronize before starting timer to ensure previous CUDA ops are done
            torch.cuda.synchronize()
            start_time = time.time()
            _ = detector.check_occlusion_batch(current_points_torch_gpu, hist_di_torch_gpu)
            # Synchronize again before ending timer to ensure this op is done
            torch.cuda.synchronize()
            end_time = time.time()
            torch_gpu_times.append(end_time - start_time)
        avg_torch_gpu_time = np.mean(torch_gpu_times)
        print(f"  Average time: {avg_torch_gpu_time:.6f} seconds")
    else:
        avg_torch_gpu_time = float('inf')

    # --- 4. Print Summary ---
    print("\n--- Benchmark Summary ---")
    print(f"Legacy NumPy (CPU):      {avg_legacy_time:.6f} s")
    print(f"PyTorch (CPU):           {avg_torch_cpu_time:.6f} s")
    if avg_torch_gpu_time != float('inf'):
        print(f"PyTorch (GPU):           {avg_torch_gpu_time:.6f} s")
        print("---------------------------")
        speedup_cpu = avg_legacy_time / avg_torch_cpu_time
        speedup_gpu = avg_legacy_time / avg_torch_gpu_time
        print(f"Speedup (CPU vs Legacy): {speedup_cpu:.2f}x")
        print(f"Speedup (GPU vs Legacy): {speedup_gpu:.2f}x")
        print(f"Speedup (GPU vs CPU):    {(avg_torch_cpu_time / avg_torch_gpu_time):.2f}x")
    else:
        print("---------------------------")
        speedup_cpu = avg_legacy_time / avg_torch_cpu_time
        print(f"Speedup (CPU vs Legacy): {speedup_cpu:.2f}x")
        print("(GPU test skipped as CUDA is not available)")


if __name__ == '__main__':
    # Ensure MDetector has the legacy method available for the benchmark
    if not hasattr(MDetector, 'check_occlusion_batch_legacy'):
        print("ERROR: MDetector class does not have 'check_occlusion_batch_legacy'.")
        print("Please rename the original NumPy method in 'src/core/m_detector/base.py' to run this benchmark.")
    else:
        detector, hist_di_numpy, hist_di_torch_cpu, hist_di_torch_gpu = setup_benchmark_environment()
        run_benchmark(detector, hist_di_numpy, hist_di_torch_cpu, hist_di_torch_gpu)