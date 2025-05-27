# src/core/m_detector/adaptive_epsilon_utils.py (NEW FILE)
import numpy as np
from typing import Optional, Dict, Any

def calculate_adaptive_epsilon(
    d_anchor: Optional[float],        # Depth of the point for which epsilon is being adapted, in its own sensor frame
    di_base: float,                   # The base epsilon value from the config (e.g., 0.5m)
    adaptive_config: Dict[str, Any]   # The specific adaptive config block for this epsilon
) -> float:
    """
    Calculates an adaptive epsilon based on the anchor depth and a config dictionary.

    Args:
        d_anchor: Depth of the anchor point. If None or invalid, di_base is returned.
        di_base: The base epsilon value (e.g., from config's epsilon_depth).
        adaptive_config: Dictionary containing 'enabled', 'dthr', 'kthr', 'dmax', 'dmin'.

    Returns:
        The calculated adaptive epsilon, or di_base if adaptation is disabled or d_anchor is invalid.
    """
    if not adaptive_config.get('enabled', False) or d_anchor is None or d_anchor < 0:
        return di_base

    # Get parameters from the specific config, with fallbacks to common defaults if not present
    dthr = adaptive_config.get('dthr', 30.0)
    kthr = adaptive_config.get('kthr', 0.05)
    # Ensure dmax and dmin are sensible relative to di_base if not perfectly configured
    dmax_default = max(di_base * 4, di_base + 1.0) # Ensure dmax is at least di_base + 1 (or 4x)
    dmax = adaptive_config.get('dmax', dmax_default)
    
    dmin_default = min(di_base * 0.2, max(0.01, di_base - 0.5)) # Ensure dmin is small but positive
    dmin = adaptive_config.get('dmin', dmin_default)


    # Core adaptive formula: di_base + kthr * (d_anchor - dthr)
    # This formula allows kthr to be negative for decreasing epsilon.
    
    adjustment = 0.0
    if d_anchor > dthr:
        adjustment = kthr * (d_anchor - dthr)
    # If kthr is positive, and d_anchor <= dthr, adjustment is 0 (or negative if max(0,...) wasn't used, but we want it to only increase if d_anchor > dthr)
    # If kthr is negative, and d_anchor <= dthr, adjustment is 0 (we only want it to decrease if d_anchor > dthr)
    
    # Refined logic based on kthr sign for clarity
    if kthr >= 0: # Epsilon increases (or stays same) for d_anchor > dthr
        if d_anchor > dthr:
            calculated_epsilon = di_base + adjustment
        else:
            calculated_epsilon = di_base
    else: # kthr < 0, Epsilon decreases for d_anchor > dthr
        if d_anchor > dthr:
            calculated_epsilon = di_base + adjustment # adjustment is negative here
        else:
            calculated_epsilon = di_base # No decrease if not beyond dthr

    # Apply min/max caps
    final_epsilon = np.clip(calculated_epsilon, dmin, dmax)
    
    return float(final_epsilon)