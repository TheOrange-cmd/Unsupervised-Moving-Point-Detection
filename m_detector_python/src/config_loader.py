# src/config_loader.py
import yaml
from typing import Dict, Any, List, Optional

class MDetectorConfigAccessor:
    def __init__(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self._config_data: Dict[str, Any] = yaml.safe_load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")
        
        if not isinstance(self._config_data, dict):
            raise ValueError(f"Configuration file {config_path} did not load as a dictionary.")

    # --- Top-Level Sections ---
    def get_nuscenes_params(self) -> Dict[str, Any]:
        return self._config_data.get('nuscenes', {})

    def get_processing_settings(self) -> Dict[str, Any]:
        settings = self._config_data.get('processing_settings', {})
        return settings
        
    def get_mdetector_output_paths(self) -> Dict[str, Any]:
        return self._config_data.get('mdetector_output_paths', {})

    def get_validation_params(self) -> Dict[str, Any]:
        return self._config_data.get('validation_params', {})

    # --- M-Detector Core Algorithm Parameters ---
    def _get_m_detector_base(self) -> Dict[str, Any]:
        return self._config_data.get('m_detector', {})

    def get_point_pre_filtering_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('point_pre_filtering', {})

    def get_depth_image_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('depth_image', {})

    def get_occlusion_determination_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('occlusion_determination', {})

    def get_event_detection_logic_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('event_detection_logic', {})
    
    def get_test4_perpendicular_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params().get('test4_perpendicular', {})
        
    def get_test2_parallel_away_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params().get('test2_parallel_away', {})

    def get_test3_parallel_towards_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params().get('test3_parallel_towards', {})

    def get_map_consistency_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('map_consistency', {})

    def get_frame_refinement_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('frame_refinement', {})
        
    def get_clustering_params(self) -> Dict[str, Any]: # Helper for refinement
        return self.get_frame_refinement_params().get('clustering', {})

    def get_region_growth_params(self) -> Dict[str, Any]: # Helper for refinement
        return self.get_frame_refinement_params().get('region_growth', {})

    def get_initialization_phase_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('initialization_phase', {})
        
    def get_bidirectional_aggregation_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('bidirectional_aggregation', {})
    
    def get_ransac_ground_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base().get('ransac_ground_params', {})
    
    def get_adaptive_epsilon_config_for_occlusion_depth(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for the main occlusion_determination.epsilon_depth."""
        return self.get_occlusion_determination_params().get('adaptive_epsilon_depth_config', {})

    def get_adaptive_epsilon_config_for_map_consistency_forward(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for map_consistency.epsilon_depth_forward_m."""
        return self.get_map_consistency_params().get('adaptive_epsilon_forward_config', {})

    def get_adaptive_epsilon_config_for_map_consistency_backward(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for map_consistency.epsilon_depth_backward_m."""
        return self.get_map_consistency_params().get('adaptive_epsilon_backward_config', {})
    
    # --- Visualization ---
    def get_visualization_params(self) -> Dict[str, Any]:
        return self._config_data.get('visualization', {})

    def get_video_generation_params(self) -> Dict[str, Any]:
        return self.get_visualization_params().get('video_generation', {})
        
    def get_k3d_plot_params(self) -> Dict[str, Any]:
        return self.get_visualization_params().get('k3d_plot', {})

    # --- Direct accessors ---
    def get_library_size(self, default: int = 20) -> int:
        di_params = self.get_depth_image_params()
        return di_params.get('library_size', default)
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Returns the entire configuration dictionary as loaded from the YAML file.
        """
        return self._config_data