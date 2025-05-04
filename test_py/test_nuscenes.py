# test_py/test_bindings.py
import sys
import os
import numpy as np
import time
import pytest
import yaml
import logging # Import logging
from conftest import create_dummy_scan
import mpy_detector as mdet

# --- Configure logging ---
# Basic config, shows INFO level and above. Use level=logging.DEBUG for more detail.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Get a logger for this module

# --- Add NuScenes Devkit to Path ---
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError as e:
    log.warning(f"nuScenes-devkit not found or import failed: {e}. Real data loading test will be skipped.")
    NUSCENES_AVAILABLE = False
except Exception as e:
    log.warning(f"An unexpected error occurred importing NuScenes: {e}")
    NUSCENES_AVAILABLE = False


# # --- Determine Module Path ---
# log.info(f"Original filepath: {__file__}")
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# build_dir = os.path.join(repo_root, 'build')
# try:
#     import mpy_detector as mdet
#     log.info("Successfully imported mpy_detector")
# except ImportError as e:
#     log.error(f"Error importing mpy_detector from default paths: {e}")
#     if build_dir not in sys.path:
#          log.info(f"Adding build directory to path: {build_dir}")
#          sys.path.insert(0, build_dir)
#          try:
#              import mpy_detector as mdet
#              log.info("Successfully imported mpy_detector after adding build dir to path.")
#          except ImportError as e2:
#              log.critical(f"Still failed to import mpy_detector after adding build dir: {e2}")
#              sys.exit(1)
#     else:
#         sys.exit(1)

# --- Test with Real NuScenes Data using add_scan ---
@pytest.mark.skipif(not NUSCENES_AVAILABLE, reason="nuScenes-devkit not available")
def test_add_scan_nuscenes_data(initialized_filter, loaded_py_config, config_path):
    """Tests the add_scan pipeline with a real nuScenes scan."""
    log.info("Running test_add_scan_nuscenes_data...")

    full_config = {}
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
    except Exception as e:
         pytest.fail(f"Failed to reload full config {config_path} for dataset section: {e}")

    if 'dataset' not in full_config or not isinstance(full_config['dataset'], dict):
        pytest.skip("Skipping nuScenes test: 'dataset' section missing or invalid in config YAML.")

    dataset_config = full_config['dataset']
    nuscenes_version = dataset_config.get('version', 'v1.0-mini')
    dataroot_raw = dataset_config.get('dataroot')

    if not dataroot_raw:
        pytest.skip("Skipping nuScenes test: 'dataroot' missing in 'dataset' config section.")

    try:
        nuscenes_dataroot = os.path.expanduser(dataroot_raw)
        log.info(f"  Using nuScenes version: {nuscenes_version}")
        log.info(f"  Expanded dataroot: {nuscenes_dataroot}")
    except Exception as e:
        pytest.fail(f"Error expanding dataroot path '{dataroot_raw}': {e}")

    if not os.path.exists(nuscenes_dataroot):
        pytest.skip(f"Skipping nuScenes test: Expanded dataroot does not exist: {nuscenes_dataroot}")

    filt = initialized_filter

    try:
        log.info(f"  Loading NuScenes ({nuscenes_version} from {nuscenes_dataroot})...")
        nusc = NuScenes(version=nuscenes_version, dataroot=nuscenes_dataroot, verbose=False)

        first_sample_token = nusc.sample[0]['token']
        lidar_top_data_token = nusc.get('sample', first_sample_token)['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_top_data_token)

        lidar_filepath = nusc.get_sample_data_path(lidar_top_data_token)
        log.info(f"  Loading point cloud from: {lidar_filepath}")
        pc = LidarPointCloud.from_file(lidar_filepath)
        points_lidar_frame = pc.points.T[:, :4]

        sensor_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_quat = Quaternion(pose_record['rotation'])
        sensor_calib_quat = Quaternion(sensor_record['rotation'])
        final_quat = ego_quat * sensor_calib_quat
        ego_translation = np.array(pose_record['translation'])
        sensor_calib_translation = np.array(sensor_record['translation'])
        final_translation = ego_translation + ego_quat.rotate(sensor_calib_translation)
        rotation_matrix = final_quat.rotation_matrix.astype(np.float64)
        position_vector = final_translation.astype(np.float64)
        timestamp_usec = lidar_data['timestamp']
        scan_timestamp_sec = timestamp_usec / 1e6

        log.info(f"  Loaded {points_lidar_frame.shape[0]} points. Timestamp: {scan_timestamp_sec:.6f}")

        points_np = points_lidar_frame.astype(np.float32)

        log.info(f"  Calling add_scan...")
        filt.add_scan(
            points_np=points_np,
            rotation=rotation_matrix,
            position=position_vector,
            timestamp=scan_timestamp_sec
        )
        log.info(f"  add_scan completed. MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")

        assert filt.get_depth_map_count() == 1, "Should have created 1 map after first nuScenes scan"
        assert filt.get_last_processed_seq_id() == 0, "Should have processed seq_id 0"
        assert filt.get_scan_buffer_size() == 1, "Scan buffer should contain 1 scan"

        # Placeholder for adding second scan logic if needed

        log.info(f"test_add_scan_nuscenes_data ({nuscenes_version}) PASSED (basic checks)")

    except ImportError:
        pytest.skip("Skipping nuScenes test due to import error previously noted.")
    except Exception as e:
        log.error(f"Error during nuScenes data processing or add_scan call: {e}", exc_info=True) # Log traceback
        pytest.fail(f"Error during nuScenes data processing or add_scan call: {e}")


# --- Placeholder Test (Keep if binding is still useful) ---
def test_placeholder_labeling_binding(initialized_filter):
    """Tests the separate placeholder labeling function binding."""
    log.info("Running test_placeholder_labeling_binding...")
    filt = initialized_filter
    num_points = 20
    points, _, _, _ = create_dummy_scan(num_points=num_points)

    try:
        log.info("Calling placeholder_labeling...")
        labels_np = filt.placeholder_labeling(points)
        log.info("placeholder_labeling completed.")
        assert isinstance(labels_np, np.ndarray)
        assert labels_np.shape == (num_points,)
        expected_labels = np.arange(num_points) % 7
        np.testing.assert_array_equal(labels_np, expected_labels)
        log.info("test_placeholder_labeling_binding PASSED")
    except Exception as e:
         log.error(f"Error calling placeholder_labeling: {e}", exc_info=True)
         pytest.fail(f"Error calling placeholder_labeling: {e}")