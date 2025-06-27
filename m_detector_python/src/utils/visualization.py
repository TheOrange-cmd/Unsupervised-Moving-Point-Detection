# src/utils/visualization.py 

import open3d as o3d
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Generator, List, Tuple

from ..core.constants import OcclusionResult

def create_lookat_matrix(eye, center, up):
    """Creates a 4x4 view matrix from eye, center, and up vectors."""
    # --- FIX #1: The Z-axis must point from the target TO the camera ---
    z_axis = eye - center
    z_axis /= np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    rotation = np.array([x_axis, y_axis, z_axis]).T
    translation = -rotation @ eye
    
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = translation
    return view_matrix

class ColorMapper:
    """
    A centralized utility for defining and applying color schemes to point clouds.
    """
    # --- Color Definitions (RGB, 0-1 range) ---
    PRED_DYNAMIC = [1.0, 0.0, 0.0]
    PRELABELED_GROUND = [0.1, 0.9, 0.1]
    INITIALIZING = [0.0, 0.7, 1.0]
    STATIC = [0.7, 0.7, 0.7]
    GT_DYNAMIC = [0.0, 0.5, 1.0]
    FILTERED_OUT = [0.4, 0.1, 0.1]
    TP = [0.0, 1.0, 0.0]
    FP = [1.0, 0.0, 0.0]
    FN = [1.0, 0.6, 0.0]
    TN = [0.5, 0.5, 0.5]

    def get_detector_output_colors(self, labels: np.ndarray) -> np.ndarray:
        colors = np.full((len(labels), 3), self.STATIC, dtype=np.float64)
        colors[labels == OcclusionResult.OCCLUDING_IMAGE.value] = self.PRED_DYNAMIC
        colors[labels == OcclusionResult.PRELABELED_STATIC_GROUND.value] = self.PRELABELED_GROUND
        return colors

    def get_gt_colors(self, gt_dynamic_mask: np.ndarray) -> np.ndarray:
        colors = np.full((len(gt_dynamic_mask), 3), self.STATIC, dtype=np.float64)
        colors[gt_dynamic_mask] = self.GT_DYNAMIC
        return colors

    def get_error_analysis_colors(self, pred_is_dyn: np.ndarray, gt_is_dyn: np.ndarray) -> np.ndarray:
        colors = np.full((len(pred_is_dyn), 3), self.TN, dtype=np.float64)
        colors[gt_is_dyn & pred_is_dyn] = self.TP
        colors[~gt_is_dyn & pred_is_dyn] = self.FP
        colors[gt_is_dyn & ~pred_is_dyn] = self.FN
        return colors

class BoxDrawer:
    """
    A utility using Open3D's built-in OrientedBoundingBox.
    """
    
    def get_box_obb(self, box, color: List[float] = [0.0, 1.0, 0.0]) -> o3d.geometry.OrientedBoundingBox:
        """Use Open3D's built-in OrientedBoundingBox - should render without warnings."""
        corners = box.corners().T
        
        # Create OrientedBoundingBox from corners
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners)
        )
        obb.color = color
        
        return obb
    
    def get_box_obb_wireframe(self, box, color: List[float] = [0.0, 1.0, 0.0]) -> o3d.geometry.LineSet:
        """Convert OrientedBoundingBox to wireframe - might still have warnings."""
        corners = box.corners().T
        
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners)
        )
        
        # Get the wireframe
        wireframe = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        wireframe.paint_uniform_color(color)
        
        return wireframe
    
class VideoRenderer:
    """Manages offscreen rendering to a video file using the MODERN Open3D API."""
    def __init__(self, output_path: str, width: int = 1920, height: int = 1080, fps: int = 20):
        self.output_path = output_path
        self.width = width
        self.height = height
        
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self.renderer.scene.set_lighting(self.renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        
        # Material for Point Clouds
        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultUnlit"
        self.pcd_material.point_size = 2.0
        
        # Material for LineSets 
        self.line_material = o3d.visualization.rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 3.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

    def render_frame(self, geometries: Dict[str, o3d.geometry.Geometry3D], camera_params: Dict[str, np.ndarray]):
        """Renders a single frame with the given geometries and camera settings."""
        self.renderer.scene.clear_geometry()
        
        for name, geom in geometries.items():
            # Apply the correct material based on the geometry's type
            if isinstance(geom, o3d.geometry.PointCloud):
                self.renderer.scene.add_geometry(name, geom, self.pcd_material)
            elif isinstance(geom, o3d.geometry.LineSet):
                # LineSet objects don't support normals - just add them directly
                self.renderer.scene.add_geometry(name, geom, self.line_material)
            
        self.renderer.setup_camera(
            60.0, # Field of view
            camera_params['center'], 
            camera_params['eye'], 
            camera_params['up']
        )
        
        img_o3d = self.renderer.render_to_image()
        img_np = np.asarray(img_o3d)
        self.video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    def close(self):
        """Releases the video writer."""
        self.video_writer.release()
        print(f"Video rendering complete. Saved to: {self.output_path}")

class InteractiveVisualizer:
    """
    Creates a standalone Open3D window for interactively visualizing a sequence of frames.
    Now supports FPS control and forward/backward stepping.
    """
    def __init__(self, window_name: str = "Interactive Analysis", width: int = 1920, height: int = 1080, fps: int = 20):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        # Check if the window was successfully created before proceeding.
        opt = self.vis.get_render_option()
        if opt is None:
            raise RuntimeError(
                "Failed to create an Open3D window. This usually means you are running in an environment "
                "without a graphical display.\n\n"
                "Common Solutions:\n"
                "1. If using SSH, connect with X11 forwarding: 'ssh -X user@hostname'\n"
                "2. If using Docker, ensure it's configured to access the host's display.\n"
                "3. If using WSL, ensure you have a running X server (like VcXsrv) or are using WSLg.\n"
            )
        
        # --- State for Animation Control ---
        self.is_paused = True
        self.advance_one_frame = False
        self.rewind_one_frame = False
        self.is_running = True
        self.geometries: Dict[str, o3d.geometry.Geometry] = {}
        
        # --- Frame Data and Indexing ---
        self.cached_frames: List[Dict[str, Any]] = []
        self.current_frame_index = 0
        self.total_frames = 0
        
        # --- FPS Control ---
        self.fps = fps
        self.frame_duration = 1.0 / self.fps if self.fps > 0 else 0
        
        self._setup_scene()
        self._register_callbacks()

    def _setup_scene(self):
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 2.0

    def _toggle_pause(self, vis):
        self.is_paused = not self.is_paused
        print(f"Animation {'PAUSED' if self.is_paused else 'PLAYING'}")
        return False

    def _advance_frame(self, vis):
        if self.is_paused:
            self.advance_one_frame = True
        return False

    def _rewind_frame(self, vis):
        if self.is_paused:
            self.rewind_one_frame = True
        return False

    def _quit_visualizer(self, vis):
        self.is_running = False
        return False

    def _register_callbacks(self):
        self.vis.register_key_callback(ord(" "), self._toggle_pause)  # Spacebar
        self.vis.register_key_callback(ord("."), self._advance_frame) # Period
        self.vis.register_key_callback(ord(","), self._rewind_frame) # Comma
        self.vis.register_key_callback(ord("Q"), self._quit_visualizer)

    def _update_display(self):
        """Updates the geometries in the visualizer for the current frame index."""
        frame_data = self.cached_frames[self.current_frame_index]
        is_first_frame = (self.current_frame_index == 0)
        
        # This is more efficient than remove/add for existing geometries
        for name, geom in frame_data['geometries'].items():
            if name not in self.geometries:
                self.vis.add_geometry(geom, reset_bounding_box=is_first_frame)
                self.geometries[name] = geom
            else:
                self.geometries[name].points = geom.points
                self.geometries[name].colors = geom.colors
                self.vis.update_geometry(self.geometries[name])
        
        # Remove geometries that are not in the current frame
        for name in list(self.geometries.keys()):
            if name not in frame_data['geometries']:
                self.vis.remove_geometry(self.geometries[name], reset_bounding_box=False)
                del self.geometries[name]
        
        self.vis.get_view_control().set_constant_z_far(200) # Prevents clipping

    def run_animation_loop(self, cached_frames: List[Dict[str, Any]]):
        self.cached_frames = cached_frames
        self.total_frames = len(cached_frames)
        if self.total_frames == 0:
            print("No frames to display.")
            return

        print("\n--- Interactive Visualizer Controls ---")
        print("  [Spacebar]: Play/Pause Animation")
        print("  [.]:        Advance one frame forward (when paused)")
        print("  [,]:        Go back one frame (when paused)")
        print("  [Q]:        Quit")
        print("------------------------------------")
        print("Animation PAUSED. Press Spacebar to start.")

        self._update_display() # Show the first frame

        while self.is_running:
            frame_start_time = time.time()

            # Handle frame advancement/rewind logic
            index_changed = False
            if self.advance_one_frame:
                self.current_frame_index = min(self.current_frame_index + 1, self.total_frames - 1)
                self.advance_one_frame = False
                index_changed = True
            elif self.rewind_one_frame:
                self.current_frame_index = max(self.current_frame_index - 1, 0)
                self.rewind_one_frame = False
                index_changed = True
            elif not self.is_paused:
                if self.current_frame_index < self.total_frames - 1:
                    self.current_frame_index += 1
                    index_changed = True
                else:
                    self.is_paused = True # Pause at the end
                    print("End of sequence. Paused.")

            if index_changed:
                self._update_display()

            # Poll events and update renderer
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()
            
            # Enforce FPS
            elapsed = time.time() - frame_start_time
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0 and not self.is_paused:
                time.sleep(sleep_time)
        
        self.vis.destroy_window()