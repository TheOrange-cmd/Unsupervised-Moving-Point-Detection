# src/config_loader.py
import yaml
from typing import Dict, Any, List, Optional

class MDetectorConfigAccessor:
    def __init__(self, config_path: str = None, config_dict: dict = None):
        """Initializes the accessor from either a file path or a dictionary."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self._config_data: Dict[str, Any] = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")
        elif config_dict:
            # Directly use the provided dictionary
            self._config_data = config_dict
        else:
            raise ValueError("MDetectorConfigAccessor must be initialized with either a config_path or a config_dict.")
        
        if not isinstance(self._config_data, dict):
            raise ValueError(f"Configuration data did not load as a dictionary.")

    # --- Top-Level Sections ---
    def get_nuscenes_params(self) -> Dict[str, Any]:
        return self._config_data['nuscenes']

    def get_processing_settings(self) -> Dict[str, Any]:
        settings = self._config_data['processing_settings']
        return settings
    
    def get_random_seed(self) -> Optional[int]:
        """Retrieves the random seed from the processing settings, if it exists."""
        return self.get_processing_settings().get('random_seed')
        
    def get_mdetector_output_paths(self) -> Dict[str, Any]:
        return self._config_data['mdetector_output_paths']

    def get_validation_params(self) -> dict:
        """
        Constructs the complete, consistent dictionary of parameters needed for metrics calculation.
        This serves as the single source of truth for validation parameters.
        """
        filt_cfg = self.get_point_pre_filtering_params()
        val_cfg = self._config_data['validation_params']
        
        # We assume a constant dynamic label value as per the algorithm's design.
        # In a future version, this could also be read from the config if needed.
        mdet_dynamic_label_value = 0 

        eval_params = {
            "mdet_label_field_name": "mdet_label",
            "mdet_dynamic_label_value": mdet_dynamic_label_value,
            "coordinate_tolerance_for_verification": 1e-3,
            "mdet_min_point_range_meters": filt_cfg['min_range_meters'],
            "mdet_max_point_range_meters": filt_cfg['max_range_meters'],
            "evaluate_only_keyframes": False, # Assuming this is a constant for now
            "gt_velocity_threshold": val_cfg['gt_velocity_threshold']
        }
        return eval_params

    # --- M-Detector Core Algorithm Parameters ---
    def _get_m_detector_base(self) -> Dict[str, Any]:
        return self._config_data['m_detector']

    def get_point_pre_filtering_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['point_pre_filtering']

    def get_depth_image_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['depth_image']

    def get_occlusion_determination_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['occlusion_determination']

    def get_event_detection_logic_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['event_detection_logic']
    
    def get_test4_perpendicular_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params()['test4_perpendicular']
        
    def get_test2_parallel_away_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params()['test2_parallel_away']

    def get_test3_parallel_towards_params(self) -> Dict[str, Any]:
        return self.get_event_detection_logic_params()['test3_parallel_towards']

    def get_map_consistency_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['map_consistency']

    def get_frame_refinement_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['frame_refinement']
        
    def get_clustering_params(self) -> Dict[str, Any]: # Helper for refinement
        return self.get_frame_refinement_params()['clustering']

    def get_region_growth_params(self) -> Dict[str, Any]: # Helper for refinement
        return self.get_frame_refinement_params()['region_growth']

    def get_initialization_phase_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['initialization_phase']
        
    def get_bidirectional_aggregation_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['bidirectional_aggregation']
    
    def get_ransac_ground_params(self) -> Dict[str, Any]:
        return self._get_m_detector_base()['ransac_ground_params']
    
    def get_adaptive_epsilon_config_for_occlusion_depth(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for the main occlusion_determination.epsilon_depth."""
        return self.get_occlusion_determination_params()['adaptive_epsilon_depth_config']

    def get_adaptive_epsilon_config_for_map_consistency_forward(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for map_consistency.epsilon_depth_forward_m."""
        return self.get_map_consistency_params()['adaptive_epsilon_forward_config']

    def get_adaptive_epsilon_config_for_map_consistency_backward(self) -> Dict[str, Any]:
        """Fetches the adaptive epsilon config for map_consistency.epsilon_depth_backward_m."""
        return self.get_map_consistency_params()['adaptive_epsilon_backward_config']
    
    # --- Visualization ---
    def get_visualization_params(self) -> Dict[str, Any]:
        return self._config_data['visualization']

    def get_video_generation_params(self) -> Dict[str, Any]:
        return self.get_visualization_params()['video_generation']
        
    def get_k3d_plot_params(self) -> Dict[str, Any]:
        return self.get_visualization_params()['k3d_plot']

    # --- Direct accessors ---
    def get_library_size(self, default: int = 20) -> int:
        di_params = self.get_depth_image_params()
        return di_params['library_size']
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Returns the entire configuration dictionary as loaded from the YAML file.
        """
        return self._config_data