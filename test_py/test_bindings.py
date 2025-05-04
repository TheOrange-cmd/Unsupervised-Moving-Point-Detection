# test_py/test_bindings.py
import sys
import os
import numpy as np
import time
import pytest
import yaml

# --- Add NuScenes Devkit to Path (adjust path as needed) ---
# You might need to install it: pip install nuscenes-devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion # For pose handling
    NUSCENES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: nuScenes-devkit not found or import failed: {e}")
    print("         Real data loading test will be skipped.")
    NUSCENES_AVAILABLE = False
except Exception as e:
    print(f"Warning: An unexpected error occurred importing NuScenes: {e}")
    NUSCENES_AVAILABLE = False


# --- Determine Module Path ---
# (Keep this section as is, ensure it finds mpy_detector)
print(f"Original filepath: {__file__}")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_dir = os.path.join(repo_root, 'build')
try:
    import mpy_detector as mdet
    print("Successfully imported mpy_detector")
except ImportError as e:
    print(f"Error importing mpy_detector: {e}")
    sys.exit(1)


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def repo_root_dir():
    """Provides the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="module")
def config_path(repo_root_dir):
    """Provides the path to the test configuration file."""
    path = os.path.join(repo_root_dir, 'test_data', 'configs', 'test_full_config.yaml')
    if not os.path.exists(path):
        # If you want to auto-create a dummy, do it here, but it's better
        # to ensure the file exists with necessary sections.
        pytest.fail(f"Test config file not found at: {path}")
    print(f"Using config path: {path}")
    return path

@pytest.fixture(scope="module")
def loaded_py_config(config_path):
    """Loads the entire YAML config file into a Python dictionary."""
    print(f"\nFixture: Loading full YAML config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            pytest.fail(f"Fixture: YAML file is empty or invalid: {config_path}")
        print("Fixture: Full YAML config loaded successfully.")
        return config_data
    except Exception as e:
        pytest.fail(f"Fixture: Error loading YAML file {config_path}: {e}")

@pytest.fixture(scope="function")
def initialized_filter(config_path):
    """Provides an initialized DynObjFilter instance using the config path."""
    # This fixture remains unchanged - C++ loads its part from the path
    print("\nFixture: Creating DynObjFilter via constructor...")
    try:
        filt = mdet.DynObjFilter(config_path=config_path)
        print("Fixture: Filter constructed successfully.")
        yield filt
        print("Fixture: DynObjFilter instance teardown.")
    except Exception as e:
        pytest.fail(f"Fixture: Error constructing DynObjFilter: {e}")

# --- Test Functions ---

def test_enum_access():
    """Tests accessing the exposed enums."""
    # (Keep this test as is)
    print("\nRunning test_enum_access...")
    assert mdet.DynObjFlg.STATIC.value == 0
    assert mdet.DynObjFlg.APPEARING.value == 1
    assert mdet.DynObjFlg.INVALID.value == 6
    print("test_enum_access PASSED")

def test_placeholder_labeling_mock_data(initialized_filter):
    """Tests calling the placeholder method with simple mock data."""
    print("\nRunning test_placeholder_labeling_mock_data...")
    dyn_filter = initialized_filter

    # Create Minimal Mock Data (5 points)
    points_np = np.array([
        [10.0, 1.0, 0.0, 70.0], # Index 0 -> STATIC (0)
        [11.0, 1.1, 0.1, 80.0], # Index 1 -> APPEARING  (1)
        [12.0, 1.2, 0.2, 90.0], # Index 2 -> OCCLUDING  (2)
        [ 5.0,-2.0, 0.5, 60.0], # Index 3 -> DISOCCLUDED  (3)
        [ 6.0,-2.1, 0.6, 50.0], # Index 4 -> SELF   (4)
    ], dtype=np.float32)
    rotation = np.identity(3, dtype=np.float64)
    position = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    scan_timestamp = time.time()

    try:
        print(f"  Calling process_scan_placeholder with {points_np.shape[0]} mock points...")
        labels_np = dyn_filter.process_scan_placeholder(
            points_np=points_np,
            rotation=rotation,
            position=position,
            timestamp=scan_timestamp
        )
        print(f"  process_scan_placeholder returned: {labels_np}")

        # --- Verification ---
        assert isinstance(labels_np, np.ndarray), "Result should be a NumPy array"
        assert labels_np.shape == (points_np.shape[0],), f"Expected shape ({points_np.shape[0]},), got {labels_np.shape}"
        assert labels_np.dtype == np.int32 or labels_np.dtype == np.int64, f"Expected dtype int32/64, got {labels_np.dtype}" # Pybind might use int64

        # Check specific label values based on index % 7
        expected_labels = np.array([i % 7 for i in range(points_np.shape[0])], dtype=labels_np.dtype)
        np.testing.assert_array_equal(labels_np, expected_labels, "Labels do not match expected modulo 7 pattern")

        print("test_placeholder_labeling_mock_data PASSED")
    except Exception as e:
        pytest.fail(f"Error calling process_scan_placeholder with mock data: {e}")


# --- Test with Real NuScenes Data (Optional, requires dataset) ---
@pytest.mark.skipif(not NUSCENES_AVAILABLE, reason="nuScenes-devkit not available")
def test_placeholder_labeling_nuscenes_data(initialized_filter, loaded_py_config):
    """Tests the placeholder pipeline with a real nuScenes scan configured via YAML."""
    print(f"\nRunning test_placeholder_labeling_nuscenes_data...")

    # --- Get dataset info from loaded Python config ---
    if 'dataset' not in loaded_py_config or not isinstance(loaded_py_config['dataset'], dict):
        pytest.skip("Skipping nuScenes test: 'dataset' section missing or invalid in config YAML.")

    dataset_config = loaded_py_config['dataset']
    nuscenes_version = dataset_config.get('version')
    dataroot_raw = dataset_config.get('dataroot')

    if not nuscenes_version or not dataroot_raw:
        pytest.skip("Skipping nuScenes test: 'version' or 'dataroot' missing in 'dataset' config section.")

    # --- Expand path ---
    try:
        nuscenes_dataroot = os.path.expanduser(dataroot_raw)
        # Optionally add expandvars if needed:
        # nuscenes_dataroot = os.path.expandvars(nuscenes_dataroot)
        print(f"  Using nuScenes version: {nuscenes_version}")
        print(f"  Raw dataroot from config: {dataroot_raw}")
        print(f"  Expanded dataroot: {nuscenes_dataroot}")
    except Exception as e:
        pytest.fail(f"Error expanding dataroot path '{dataroot_raw}': {e}")

    if not os.path.exists(nuscenes_dataroot):
        pytest.skip(f"Skipping nuScenes test: Expanded dataroot does not exist: {nuscenes_dataroot}")

    dyn_filter = initialized_filter # Get filter instance

    try:
        print(f"  Loading NuScenes ({nuscenes_version} from {nuscenes_dataroot})...")
        nusc = NuScenes(version=nuscenes_version, dataroot=nuscenes_dataroot, verbose=False)

        # Get the first sample and its LIDAR_TOP data
        first_sample_token = nusc.sample[0]['token']
        lidar_top_data_token = nusc.get('sample', first_sample_token)['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_top_data_token)

        # Load point cloud
        lidar_filepath = nusc.get_sample_data_path(lidar_top_data_token)
        print(f"  Loading point cloud from: {lidar_filepath}")
        pc = LidarPointCloud.from_file(lidar_filepath)
        points_lidar_frame = pc.points.T[:, :4] # Get Nx4 (x, y, z, intensity)

        # Get pose information (sensor pose in global frame)
        sensor_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Combine sensor calibration and ego pose to get sensor pose in global frame
        # Sensor pose = Global Ego Pose * Sensor Calibration Pose
        ego_quat = Quaternion(pose_record['rotation'])
        sensor_calib_quat = Quaternion(sensor_record['rotation'])
        final_quat = ego_quat * sensor_calib_quat # Order matters!

        ego_translation = np.array(pose_record['translation'])
        sensor_calib_translation = np.array(sensor_record['translation'])
        # Rotate sensor translation to global frame and add ego translation
        final_translation = ego_translation + ego_quat.rotate(sensor_calib_translation)

        rotation_matrix = final_quat.rotation_matrix.astype(np.float64)
        position_vector = final_translation.astype(np.float64)
        timestamp_usec = lidar_data['timestamp']
        scan_timestamp_sec = timestamp_usec / 1e6 # Convert microseconds to seconds

        print(f"  Loaded {points_lidar_frame.shape[0]} points. Timestamp: {scan_timestamp_sec:.6f}")
        print(f"  Sensor Global Pos: {position_vector}")

        # Ensure data types are correct for C++ bindings
        points_np = points_lidar_frame.astype(np.float32)

        # Call the placeholder processing
        print(f"  Calling process_scan_placeholder...")
        labels_np = dyn_filter.process_scan_placeholder(
            points_np=points_np,
            rotation=rotation_matrix,
            position=position_vector,
            timestamp=scan_timestamp_sec
        )
        print(f"  process_scan_placeholder returned {labels_np.shape[0]} labels.")

        # --- Verification ---
        assert isinstance(labels_np, np.ndarray)
        assert labels_np.shape == (points_np.shape[0],)
        assert labels_np.dtype == np.int32 or labels_np.dtype == np.int64

        expected_labels = np.array([i % 7 for i in range(points_np.shape[0])], dtype=labels_np.dtype)
        np.testing.assert_array_equal(labels_np, expected_labels, "Labels do not match expected modulo 7 pattern for nuScenes data")

        print(f"test_placeholder_labeling_nuscenes_data ({nuscenes_version}) PASSED")

    except ImportError:
        pytest.skip("Skipping nuScenes test due to import error previously noted.")
    except Exception as e:
        pytest.fail(f"Error during nuScenes data processing or placeholder call: {e}")