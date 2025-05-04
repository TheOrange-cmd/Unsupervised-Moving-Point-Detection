# test_labeling_appearing.py
import pytest
import numpy as np
import mpy_detector as mdet # Assuming mpy_detector is importable
import logging
import os

log = logging.getLogger(__name__)

# --- Helper to create specific scans ---
def create_appearing_scan(num_bg=5, num_app=1, timestamp=0.0, appearing_now=False):
    """Creates a scan with background and optional appearing points."""
    total_points = num_bg + (num_app if appearing_now else 0)
    points = np.zeros((total_points, 4), dtype=np.float32)
    points[:, 3] = 70.0 # Intensity

    # Background points (far away)
    points[:num_bg, 0] = np.linspace(50, 60, num_bg) # X
    points[:num_bg, 1] = np.linspace(-5, 5, num_bg)  # Y
    points[:num_bg, 2] = 1.0                         # Z

    # Appearing points (closer, only if appearing_now is True)
    if appearing_now:
        points[num_bg:, 0] = np.linspace(5, 6, num_app)    # Closer X
        points[num_bg:, 1] = np.linspace(0.5, 0.6, num_app) # Y (avoid self-filter region)
        points[num_bg:, 2] = 0.5                           # Z

    rotation = np.identity(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    return points, rotation, position, timestamp

# --- Test Function ---

def test_appearing_points(initialized_filter, loaded_py_config):
    """
    Tests if points appearing after several maps are labeled APPEARING.
    Requires occluded_map_thr1 >= 2 in the config.
    """
    log.info("Running test_appearing_points...")
    filt = initialized_filter

    # --- Get relevant config params ---
    try:
        depth_map_dur = loaded_py_config['depth_map_dur']
        occluded_map_thr1 = loaded_py_config['occluded_map_thr1']
        dyn_filter_en = loaded_py_config.get('dyn_filter_en', False) # Default to False if missing
        log.info(f"Config params: depth_map_dur={depth_map_dur}, occluded_map_thr1={occluded_map_thr1}, dyn_filter_en={dyn_filter_en}")
    except KeyError as e:
        pytest.fail(f"Missing required config parameter: {e}")

    if not dyn_filter_en:
        pytest.skip("Skipping test_appearing_points: dyn_filter_en is false in config.")

    if occluded_map_thr1 < 2:
         pytest.skip(f"Skipping test_appearing_points: occluded_map_thr1 ({occluded_map_thr1}) must be >= 2 for this test logic.")

    # --- Test Scenario ---
    num_bg = 5
    num_app = 2
    appearing_point_start_idx = num_bg # Index where appearing points start in the array

    scans_to_add = []
    # Scan 0: Background only (Creates Map 0)
    scans_to_add.append(create_appearing_scan(num_bg, num_app, timestamp=1.0, appearing_now=False))
    # Scan 1: Background only (Adds to Map 0)
    scans_to_add.append(create_appearing_scan(num_bg, num_app, timestamp=1.0 + depth_map_dur*0.5, appearing_now=False))
    # Scan 2: Background only (Creates Map 1)
    scans_to_add.append(create_appearing_scan(num_bg, num_app, timestamp=1.0 + depth_map_dur + 0.01, appearing_now=False))
    # Scan 3: Background only (Adds to Map 1)
    scans_to_add.append(create_appearing_scan(num_bg, num_app, timestamp=1.0 + depth_map_dur*1.5 + 0.01, appearing_now=False))
    # Scan 4: Background + Appearing Points (Creates Map 2) - These points should be APPEARING
    scan4_ts = 1.0 + 2 * depth_map_dur + 0.02
    scan4_seq_id = 4
    scans_to_add.append(create_appearing_scan(num_bg, num_app, timestamp=scan4_ts, appearing_now=True))

    # --- Add Scans ---
    for i, (points, rot, pos, ts) in enumerate(scans_to_add):
        log.info(f"Adding Scan {i}, SeqID={i}, TS={ts:.3f}, NumPoints={len(points)}")
        filt.add_scan(points, rot, pos, ts)
        log.info(f" -> MapCount={filt.get_depth_map_count()}, LastProcID={filt.get_last_processed_seq_id()}")
        # Basic check after each add
        assert filt.get_last_processed_seq_id() == i

    # --- Get Processed Info for the critical scan ---
    log.info(f"Getting processed points info for SeqID={scan4_seq_id}")
    processed_info = filt.get_processed_points_info(scan4_seq_id)

    assert processed_info is not None, "getProcessedPointsInfo returned None"
    assert len(processed_info) == num_bg + num_app, f"Expected {num_bg + num_app} processed points, got {len(processed_info)}"

    # --- Verify Labels ---
    appearing_found = 0
    static_found = 0
    other_labels = []

    log.info("Verifying labels...")
    for point_info in processed_info:
        log.debug(f"  Point Idx={point_info.original_index}, Label={point_info.label}, Loc=({point_info.local_x:.2f}, {point_info.local_y:.2f}, {point_info.local_z:.2f})")
        if point_info.original_index >= appearing_point_start_idx:
            # These are the points that should be appearing
            if point_info.label == mdet.DynObjLabel.APPEARING:
                appearing_found += 1
            else:
                other_labels.append((point_info.original_index, point_info.label))
        else:
            # These are the background points
            if point_info.label == mdet.DynObjLabel.STATIC:
                static_found += 1
            else:
                 other_labels.append((point_info.original_index, point_info.label))

    log.info(f"Verification Results: Appearing={appearing_found}/{num_app}, Static={static_found}/{num_bg}, Other={other_labels}")

    assert appearing_found == num_app, f"Expected {num_app} APPEARING points, found {appearing_found}. Incorrect labels: {other_labels}"
    assert static_found == num_bg, f"Expected {num_bg} STATIC points, found {static_found}. Incorrect labels: {other_labels}"

    log.info("test_appearing_points PASSED")