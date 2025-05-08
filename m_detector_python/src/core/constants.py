"""
Core constants used throughout the M-Detector project, such as Enums.
"""

# core/constants.py

from enum import Enum

class OcclusionResult(Enum):
    """
    Enum for the result of occlusion checks, indicating the relationship
    of a point from a 'current' scan with respect to data in a 'reference' (e.g., historical) scan.
    """
    OCCLUDING_IMAGE = 0   # Current point is in front of data in the reference image (suggests current point is dynamic).
    OCCLUDED_BY_IMAGE = 1 # Current point is behind data in the reference image (suggests current point is static).
    EMPTY_IN_IMAGE = 2    # Current point projects to an area with no data in the reference image (ambiguous).
    UNDETERMINED = 3      # Cannot determine the occlusion relationship (e.g., point outside FoV, or within epsilon thresholds).