# test_py/test_bindings.py
import sys
import os
import numpy as np
import time
import pytest
import yaml
import logging # Import logging
from conftest import create_dummy_scan

# --- Configure logging ---
# Basic config, shows INFO level and above. Use level=logging.DEBUG for more detail.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Get a logger for this module

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

# --- Test Functions ---

def test_enum_access():
    """Tests accessing the exposed enums."""
    log.info("Running test_enum_access...")
    # Use the correct Python binding name 'DynObjLabel'
    assert hasattr(mdet, "DynObjLabel"), "Module 'mdet' should have attribute 'DynObjLabel'"
    assert mdet.DynObjLabel.STATIC.value == 0
    assert mdet.DynObjLabel.APPEARING.value == 1
    assert mdet.DynObjLabel.OCCLUDING.value == 2
    assert mdet.DynObjLabel.DISOCCLUDED.value == 3
    assert mdet.DynObjLabel.SELF.value == 4
    assert mdet.DynObjLabel.UNCERTAIN.value == 5
    assert mdet.DynObjLabel.INVALID.value == 6
    log.info("test_enum_access PASSED")


def test_filter_initialization(initialized_filter, loaded_py_config):
    """Tests the initial state of the filter after construction."""
    log.info("Running test_filter_initialization...")
    filt = initialized_filter
    assert filt is not None
    assert filt.get_depth_map_count() == 0
    assert filt.get_last_processed_seq_id() == np.uint64(-1) # Or UINT64_MAX
    assert filt.get_scan_buffer_size() == 0
    expected_capacity = loaded_py_config.get('history_length')
    assert expected_capacity is not None, "'history_length' not found in loaded config"
    assert filt.get_scan_buffer_capacity() == expected_capacity
    log.info("test_filter_initialization PASSED")


def test_add_first_scan(initialized_filter):
    """Tests adding the very first scan using add_scan."""
    log.info("Running test_add_first_scan...")
    filt = initialized_filter
    points, rot, pos, ts = create_dummy_scan(timestamp=1.0)

    log.info(f"Calling add_scan with ts={ts}")
    filt.add_scan(points, rot, pos, ts)
    log.info(f"add_scan completed. MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")

    assert filt.get_depth_map_count() == 1, "Should create the first depth map"
    assert filt.get_last_processed_seq_id() == 0, "Seq ID of the first processed frame should be 0"
    assert filt.get_scan_buffer_size() == 1
    log.info("test_add_first_scan PASSED")


def test_add_multiple_scans_within_duration(initialized_filter, loaded_py_config):
    """Tests adding scans close in time (should use the same depth map)."""
    log.info("Running test_add_multiple_scans_within_duration...")
    filt = initialized_filter
    depth_map_dur = loaded_py_config.get('depth_map_dur')
    assert depth_map_dur is not None, "'depth_map_dur' not found in loaded config"

    ts_start = 1.0
    num_scans = 3
    time_step = depth_map_dur / (num_scans + 1) # Ensure steps are smaller than duration

    for i in range(num_scans):
        ts = ts_start + i * time_step
        points, rot, pos, _ = create_dummy_scan(timestamp=ts)
        log.info(f"Calling add_scan for scan {i}, ts={ts:.3f}")
        filt.add_scan(points, rot, pos, ts)
        log.info(f" Added scan {i}, ts={ts:.3f}, MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")

    assert filt.get_depth_map_count() == 1, "Should still only have 1 map"
    assert filt.get_last_processed_seq_id() == num_scans - 1, f"Last processed ID should be {num_scans - 1}"
    assert filt.get_scan_buffer_size() == num_scans
    log.info("test_add_multiple_scans_within_duration PASSED")


def test_add_scans_trigger_new_map(initialized_filter, loaded_py_config):
    """Tests adding a scan far enough in time to trigger a new depth map."""
    log.info("Running test_add_scans_trigger_new_map...")
    filt = initialized_filter
    depth_map_dur = loaded_py_config.get('depth_map_dur')
    # frame_dur = loaded_py_config.get('frame_dur', 0.1) # Not strictly needed for this test logic
    assert depth_map_dur is not None, "'depth_map_dur' not found in loaded config"
    assert depth_map_dur > 0, "depth_map_dur must be positive"

    ts1 = 1.0
    # Ensure ts2 is *at least* depth_map_dur after ts1
    # Add a small epsilon to avoid floating point boundary issues
    ts2 = ts1 + depth_map_dur + 0.001

    points1, rot1, pos1, _ = create_dummy_scan(timestamp=ts1)
    points2, rot2, pos2, _ = create_dummy_scan(timestamp=ts2)

    log.info(f"Calling add_scan for scan 0, ts={ts1:.3f}")
    filt.add_scan(points1, rot1, pos1, ts1)
    log.info(f" Added scan 0, ts={ts1:.3f}, MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")
    assert filt.get_depth_map_count() == 1, "Should have 1 map after first scan"
    assert filt.get_last_processed_seq_id() == 0

    log.info(f"Calling add_scan for scan 1, ts={ts2:.3f} (diff={ts2-ts1:.3f} vs dur={depth_map_dur:.3f})")
    filt.add_scan(points2, rot2, pos2, ts2)
    log.info(f" Added scan 1, ts={ts2:.3f}, MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")
    # Now the time difference should be >= depth_map_dur
    assert filt.get_depth_map_count() == 2, "Should have created a second map"
    assert filt.get_last_processed_seq_id() == 1


def test_add_scans_reach_max_maps(initialized_filter, loaded_py_config):
    """Tests adding scans until max_depth_map_num is reached and rotation occurs."""
    log.info("Running test_add_scans_reach_max_maps...")
    filt = initialized_filter
    max_maps = loaded_py_config.get('max_depth_map_num')
    history_len = loaded_py_config.get('history_length')
    depth_map_dur = loaded_py_config.get('depth_map_dur')
    assert max_maps is not None, "'max_depth_map_num' not found"
    assert history_len is not None, "'history_length' not found"
    assert depth_map_dur is not None, "'depth_map_dur' not found"

    num_scans_to_add = max_maps + 2 # Add enough to fill and rotate
    ts_increment = depth_map_dur + 0.1 # Ensure each scan triggers new map

    current_ts = 1.0
    for i in range(num_scans_to_add):
        points, rot, pos, _ = create_dummy_scan(timestamp=current_ts)
        log.info(f"Calling add_scan for scan {i}, ts={current_ts:.1f}")
        filt.add_scan(points, rot, pos, current_ts)
        log.info(f" Added scan {i}, ts={current_ts:.1f}, MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}, BufferSize={filt.get_scan_buffer_size()}")

        if i < max_maps:
            assert filt.get_depth_map_count() == i + 1, f"Scan {i}: Expected {i+1} maps"
        else:
            assert filt.get_depth_map_count() == max_maps, f"Scan {i}: Expected map count to cap at {max_maps}"

        assert filt.get_last_processed_seq_id() == i, f"Scan {i}: Expected last processed ID to be {i}"

        current_ts += ts_increment

    assert filt.get_depth_map_count() == max_maps, f"Final map count should be {max_maps}"
    assert filt.get_last_processed_seq_id() == num_scans_to_add - 1
    assert filt.get_scan_buffer_size() == min(num_scans_to_add, history_len)
    log.info("test_add_scans_reach_max_maps PASSED")

