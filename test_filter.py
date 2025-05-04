import sys
import os
import numpy as np
import time # For timestamp

# --- Add build directory ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './build'))
print(f"Adding to sys.path: {module_path}")
sys.path.insert(0, module_path)

try:
    import mpy_detector as mdet
    print("Successfully imported mpy_detector")
except ImportError as e:
    print(f"Error importing mpy_detector: {e}")
    sys.exit(1)

# --- Test ---

# 1. Load Config
config_file = '/home/drugge/Unsupervised-Moving-Point-Detection/test/config/test_full_config.yaml' # Adjust path
print(f"\nLoading config from: {config_file}")
try:
    params = mdet.load_config(config_file)
    print("Config loaded successfully.")
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

# 2. Instantiate Filter
print("\nInstantiating DynObjFilter...")
try:
    dyn_filter = mdet.DynObjFilter()
    print("DynObjFilter instantiated.")
except Exception as e:
    print(f"Error instantiating DynObjFilter: {e}")
    sys.exit(1)

# 3. Initialize Filter
print("\nInitializing DynObjFilter...")
try:
    init_ok = dyn_filter.init(config_file)
    if not init_ok:
        print("Filter initialization failed!")
        sys.exit(1)
    print("Filter initialized successfully.")
except Exception as e:
    print(f"Error initializing DynObjFilter: {e}")
    sys.exit(1)

# 4. Create Minimal Mock Data
print("\nCreating mock data...")
# Create a NumPy array (5 points, 4 features: x, y, z, intensity)
# Make sure dtype is float32, as expected by numpy_to_pcl
points_np = np.array([
    [10.0, 1.0, 0.0, 70.0],
    [11.0, 1.1, 0.1, 80.0],
    [12.0, 1.2, 0.2, 90.0],
    [13.0, 1.3, 0.3, 100.0],
    [14.0, 1.4, 0.4, 110.0]
], dtype=np.float32)
print(f"Mock points shape: {points_np.shape}, dtype: {points_np.dtype}")

# Create identity rotation (Eigen::Matrix3d)
rot = np.identity(3, dtype=np.float64)
print(f"Mock rotation:\n{rot}")

# Create zero translation (Eigen::Vector3d)
pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
print(f"Mock position: {pos}")

# Create a timestamp (double)
scan_time = time.time()
print(f"Mock time: {scan_time}")

# 5. Call the Filter Stub
print("\nCalling filter stub...")
try:
    # Pass the NumPy array and Eigen types directly
    dyn_count, static_count = dyn_filter.filter(points_np, rot, pos, scan_time)
    print(f"Filter call successful.")
    print(f"  Returned Dynamic Count: {dyn_count}")
    print(f"  Returned Static Count: {static_count}")

    # Basic assertion for the stub
    assert dyn_count == 0
    assert static_count == points_np.shape[0]
    print("Counts match expected stub output.")

except Exception as e:
    print(f"Error calling filter: {e}")
    sys.exit(1)

print("\nMinimal NumPy -> PCL -> Filter Stub test completed successfully!")

# --- Next Step ---
# Now, modify the filter binding and stub to return NumPy arrays
# by implementing pcl_to_numpy and calling it in the binding lambda.
# Then, start implementing the actual filter logic.