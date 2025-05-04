# test_py/test_bindings.py
import sys
import os
import numpy as np
import time
import pytest
import yaml
import logging # Import logging

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


# --- Determine Module Path ---
log.info(f"Original filepath: {__file__}")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_dir = os.path.join(repo_root, 'build')
try:
    import mpy_detector as mdet
    log.info("Successfully imported mpy_detector")
except ImportError as e:
    log.error(f"Error importing mpy_detector from default paths: {e}")
    if build_dir not in sys.path:
         log.info(f"Adding build directory to path: {build_dir}")
         sys.path.insert(0, build_dir)
         try:
             import mpy_detector as mdet
             log.info("Successfully imported mpy_detector after adding build dir to path.")
         except ImportError as e2:
             log.critical(f"Still failed to import mpy_detector after adding build dir: {e2}")
             sys.exit(1)
    else:
        sys.exit(1)


# --- Pytest Fixtures (Using your original fixtures) ---

@pytest.fixture(scope="module")
def repo_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="module")
def config_path(repo_root_dir):
    path = os.path.join(repo_root_dir, 'test_data', 'configs', 'test_full_config.yaml')
    if not os.path.exists(path):
        pytest.fail(f"Test config file not found at: {path}")
    log.info(f"Using config path: {path}")
    return path

@pytest.fixture(scope="module")
def loaded_py_config(config_path):
    log.info(f"Fixture: Loading full YAML config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None or 'dyn_obj' not in config_data:
            pytest.fail(f"Fixture: YAML file is empty, invalid, or missing 'dyn_obj' section: {config_path}")
        log.info("Fixture: Full YAML config loaded successfully.")
        return config_data['dyn_obj']
    except Exception as e:
        pytest.fail(f"Fixture: Error loading YAML file {config_path}: {e}")

@pytest.fixture(scope="function")
def initialized_filter(config_path):
    log.info("Fixture: Creating DynObjFilter via constructor...")
    try:
        # Assuming DynObjFilter constructor prints its own logs/errors
        filt = mdet.DynObjFilter(config_path=config_path)
        log.info("Fixture: Filter constructed successfully.")
        yield filt
        log.info("Fixture: DynObjFilter instance cleanup (if any).")
    except Exception as e:
        # Use logging for failure, pytest.fail stops execution
        log.error(f"Fixture: Error constructing DynObjFilter: {e}")
        pytest.fail(f"Fixture: Error constructing DynObjFilter: {e}")

# --- Helper Function ---
def create_dummy_scan(num_points=10, timestamp=0.0):
    points = np.zeros((num_points, 4), dtype=np.float32)
    points[:, 0] = np.linspace(1, 10, num_points)
    points[:, 3] = 1.0
    rotation = np.identity(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    return points, rotation, position, timestamp