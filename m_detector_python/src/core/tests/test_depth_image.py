# src/core/tests/test_depth_image.py

import unittest
import numpy as np
import torch
from typing import Dict, Any

# Import both the legacy and the new (to be implemented) DepthImage classes
from src.core.depth_image_legacy import DepthImage as DepthImageLegacy
from src.core.depth_image import DepthImage as DepthImageTorch

# It's good practice to install and use torch-scatter for the refactor
try:
    from torch_scatter import scatter_min, scatter_max, scatter_add
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


class TestDepthImageRefactor(unittest.TestCase):
    """
    A test suite to verify that the PyTorch-based DepthImage class
    produces the same results as the original NumPy-based version.
    """

    def setUp(self):
        """This method is called before each test function to set up test data."""
        # --- 1. Define Common Test Data ---
        self.device = torch.device("cpu") # Use CPU for tests for simplicity and consistency
        self.timestamp = 123456789.0
        
        # A 4x4 identity matrix for the global pose (sensor is at the origin)
        self.image_pose_global = np.eye(4, dtype=np.float32)

        # Configuration parameters for the depth image projection
        self.depth_image_params: Dict[str, Any] = {
            'resolution_h_deg': 0.4,
            'resolution_v_deg': 1.6,
            'phi_min_rad': -np.pi,
            'phi_max_rad': np.pi,
            'theta_min_rad': np.deg2rad(-15.0),
            'theta_max_rad': np.deg2rad(15.0),
        }

        # A sample point cloud with interesting cases:
        # - Point at origin (should be invalid depth)
        # - Point within FOV
        # - Two points that project to the same pixel
        # - Point outside FOV (horizontally)
        # - Point outside FOV (vertically)
        self.points_global_batch_np = np.array([
            [0.0, 0.0, 0.0],          # Case 1: Invalid depth
            [10.0, 0.0, 1.0],         # Case 2: In FOV
            [20.0, 0.5, -2.0],        # Case 3a: Projects to a pixel
            [20.1, 0.4, -2.1],        # Case 3b: Projects to the same pixel as 3a
            [5.0, 10.0, 0.5],         # Case 4: Outside horizontal FOV (phi is large)
            [15.0, 1.0, 10.0],        # Case 5: Outside vertical FOV (theta is large)
        ], dtype=np.float32)

        # --- 2. Instantiate Both Versions of the Class ---
        self.di_numpy = DepthImageLegacy(
            self.image_pose_global, self.depth_image_params, self.timestamp
        )
        
        # This assumes you have the new DepthImage class structure in src/core/depth_image.py
        self.di_torch = DepthImageTorch(
            self.image_pose_global, self.depth_image_params, self.timestamp, self.device
        )

    def test_01_projection_batch(self):
        """
        Tests that project_points_batch produces identical results.
        This is the most critical test to pass first.
        """
        # --- Run the legacy NumPy version ---
        np_local, np_sph, np_pix, np_mask = self.di_numpy.project_points_batch(
            self.points_global_batch_np
        )

        # --- Run the new PyTorch version ---
        points_torch = torch.from_numpy(self.points_global_batch_np).to(self.device)
        torch_local, torch_sph, torch_pix, torch_mask = self.di_torch.project_points_batch(
            points_torch
        )

        # --- Compare the results ---
        # Move PyTorch results to CPU and convert to NumPy for comparison
        torch_local_np = torch_local.cpu().numpy()
        torch_sph_np = torch_sph.cpu().numpy()
        torch_pix_np = torch_pix.cpu().numpy()
        torch_mask_np = torch_mask.cpu().numpy()

        # Use np.allclose for floating point arrays due to potential tiny precision differences
        self.assertTrue(np.allclose(np_local, torch_local_np, atol=1e-6), "Local coordinates do not match")
        self.assertTrue(np.allclose(np_sph, torch_sph_np, atol=1e-6), "Spherical coordinates do not match")
        
        # Use np.array_equal for integer arrays and boolean masks
        self.assertTrue(np.array_equal(np_pix, torch_pix_np), "Pixel indices do not match")
        self.assertTrue(np.array_equal(np_mask, torch_mask_np), "Validity masks do not match")

    @unittest.skipIf(not TORCH_SCATTER_AVAILABLE, "torch-scatter is not installed, skipping add_points_batch test.")
    def test_02_add_points_and_pixel_stats(self):
        """
        Tests that add_points_batch correctly populates pixel statistics.
        This test depends on project_points_batch working correctly.
        """
        # --- Run both versions ---
        self.di_numpy.add_points_batch(self.points_global_batch_np)
        self.di_torch.add_points_batch(self.points_global_batch_np)

        # --- Compare pixel statistic grids ---
        # Convert torch tensors to numpy for comparison
        torch_min_depth_np = self.di_torch.pixel_min_depth.cpu().numpy()
        torch_max_depth_np = self.di_torch.pixel_max_depth.cpu().numpy()
        torch_count_np = self.di_torch.pixel_count.cpu().numpy()

        self.assertTrue(np.allclose(self.di_numpy.pixel_min_depth, torch_min_depth_np, equal_nan=True), "pixel_min_depth grids do not match")
        self.assertTrue(np.allclose(self.di_numpy.pixel_max_depth, torch_max_depth_np, equal_nan=True), "pixel_max_depth grids do not match")
        self.assertTrue(np.array_equal(self.di_numpy.pixel_count, torch_count_np), "pixel_count grids do not match")

        # --- Compare the pixel_original_indices dictionary ---
        # This is a dictionary of lists. To compare them robustly, we should sort the lists first.
        numpy_indices = {k: sorted(v) for k, v in self.di_numpy.pixel_original_indices.items()}
        torch_indices = {k: sorted(v) for k, v in self.di_torch.pixel_original_indices.items()}
        self.assertDictEqual(numpy_indices, torch_indices, "pixel_original_indices do not match")

    def test_03_getters(self):
        """Tests that data-retrieval getters return correct NumPy arrays."""
        self.di_numpy.add_points_batch(self.points_global_batch_np)
        self.di_torch.add_points_batch(self.points_global_batch_np)
        
        # Compare get_original_points_global
        np_pts = self.di_numpy.get_original_points_global()
        torch_pts = self.di_torch.get_original_points_global() # This getter should return a numpy array
        self.assertIsInstance(torch_pts, np.ndarray, "Getter should return a NumPy array")
        self.assertTrue(np.allclose(np_pts, torch_pts), "get_original_points_global results do not match")

        # Compare get_all_point_labels
        np_labels = self.di_numpy.get_all_point_labels()
        torch_labels = self.di_torch.get_all_point_labels()
        self.assertIsInstance(torch_labels, np.ndarray, "Getter should return a NumPy array")
        self.assertTrue(np.array_equal(np_labels, torch_labels), "get_all_point_labels results do not match")


if __name__ == '__main__':
    unittest.main()