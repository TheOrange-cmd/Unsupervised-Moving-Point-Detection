# src/core/constants.py

"""
Core constants used throughout the M-Detector project, such as Enums.
"""

from enum import Enum

class OcclusionResult(Enum):
    """
    Enum for the result of occlusion checks, indicating the relationship
    of a point from a 'current' scan with respect to data in a 'reference' scan.
    """
    # Point is in front of the reference map. The primary indicator of a dynamic point.
    OCCLUDING_IMAGE = 0
    
    # Point is behind the reference map. Strong indicator of a static point.
    OCCLUDED_BY_IMAGE = 1
    
    # Point projects to an empty area in the reference map. Ambiguous case.
    EMPTY_IN_IMAGE = 2
    
    # Occlusion status could not be determined (e.g., out of FoV, edge cases).
    UNDETERMINED = 3
    
    # Point was pre-labeled as static ground by an upstream process (e.g., RANSAC).
    # These points are excluded from dynamic checks but used for map consistency.
    PRELABELED_STATIC_GROUND = 4

# --- Centralized Algorithm Constants ---

# The integer value representing a dynamic point throughout the system.
# This is the single source of truth for what constitutes a "dynamic" label.
DYNAMIC_LABEL_VALUE = OcclusionResult.OCCLUDING_IMAGE.value