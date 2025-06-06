# src/core/m_detector/tests/test_occlusion_checks.py

import unittest
import numpy as np
import torch
import os
import yaml

# Add project root to path to allow direct imports
import sys
# This assumes your test runner is initiated from the project root.
# If not, you might need to adjust the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from src.config_loader import MDetectorConfigAccessor
from src.core.m_detector.base import MDetector
from src.core.depth_image import DepthImage as DepthImageTorch # New PyTorch version
from src.core.depth_image_legacy import DepthImage as DepthImageLegacy # Old NumPy version
from src.core.constants import OcclusionResult


class TestOcclusionChecksRefactor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a dummy config and a single MDetector instance for all tests."""
        # Create a dummy config file for the test
        cls.config_path = "temp_test_config.yaml"
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
        with open(cls.config_path, 'w') as f:
            yaml.dump(dummy_config, f)

        config_accessor = MDetectorConfigAccessor(cls.config_path)
        cls.detector = MDetector(config_accessor)
        cls.detector.device = torch.device("cpu") # Force CPU for tests
    
    @classmethod
    def tearDownClass(cls):
        """Remove the dummy config file after all tests are done."""
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)

    def test_check_occlusion_batch(self):
        """
        Tests that the refactored check_occlusion_batch produces
        the same results as the legacy NumPy version.
        """
        # --- 1. Setup: Create a historical DepthImage with known data ---
        hist_pose = np.eye(4, dtype=np.float32)
        hist_timestamp = 1000.0
        
        # Points to populate the historical depth image
        hist_points = np.array([
            [10.0, 0.0, 0.0],       # Point A: depth=10
            [10.5, 0.05, 0.05],     # Point B: depth=10.5, in same pixel as A
            [30.0, 2.0, 1.0],       # Point C: depth=30, in a different pixel
        ])

        # Create and populate both legacy and torch versions of the historical DI
        hist_di_numpy = DepthImageLegacy(hist_pose, self.detector.config_accessor.get_depth_image_params(), hist_timestamp)
        hist_di_numpy.add_points_batch(hist_points)

        hist_di_torch = DepthImageTorch(hist_pose, self.detector.config_accessor.get_depth_image_params(), hist_timestamp, self.detector.device)
        hist_di_torch.add_points_batch(hist_points)

        # --- 2. Setup: Define the current points to be checked ---
        current_points = np.array([
            [9.5, 0.0, 0.0],        # Should be OCCLUDING (in front of point A)
            [10.2, 0.02, 0.02],     # Should be UNDETERMINED (within epsilon of A/B)
            [11.0, -0.05, -0.05],   # Should be OCCLUDED_BY (behind point B)
            [5.0, -5.0, 0.0],       # Should be EMPTY_IN_IMAGE (projects to an empty area)
            [50.0, 50.0, 50.0],     # Should be UNDETERMINED (out of FOV)
        ])

        # --- 3. Run and Compare ---
        # Run legacy version (we have to rename it in the source code first)
        legacy_results_enum = self.detector.check_occlusion_batch_legacy(current_points, hist_di_numpy)
        legacy_results = np.array([res.value for res in legacy_results_enum])

        # Run new torch version
        current_points_torch = torch.from_numpy(current_points).float().to(self.detector.device)
        torch_results_tensor = self.detector.check_occlusion_batch(current_points_torch, hist_di_torch)
        torch_results = torch_results_tensor.cpu().numpy()

        # --- 4. Assert ---
        self.assertTrue(np.array_equal(legacy_results, torch_results),
                        f"Occlusion batch results do not match!\n"
                        f"Legacy: {[OcclusionResult(r).name for r in legacy_results]}\n"
                        f"Torch:  {[OcclusionResult(r).name for r in torch_results]}")

if __name__ == '__main__':
    unittest.main()