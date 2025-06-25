# src/core/m_detector/adaptive_epsilon_utils.py

import torch


def calculate_adaptive_epsilon(
    d_anchors: torch.Tensor,
    base_epsilon: float,
    enabled: bool,
    dthr: float,
    kthr: float,
    dmax: float,
    dmin: float
) -> torch.Tensor:
    """
    Calculates an adaptive epsilon value for a batch of points based on their anchor depths.
    The argument names (dthr, kthr, etc.) are intentionally matched with the keys in the 
    config file to allow for dictionary unpacking.

    Args:
        d_anchors (torch.Tensor): A 1D tensor of anchor depths for each point.
        base_epsilon (float): The base epsilon value.
        enabled (bool): Whether the adaptive logic is enabled.
        dthr (float): Depth threshold for applying the linear scaling factor.
        kthr (float): Scaling factor for depths beyond the threshold.
        dmax (float): The maximum possible epsilon value.
        dmin (float): The minimum possible epsilon value.

    Returns:
        torch.Tensor: A 1D tensor of calculated epsilon values, same size as d_anchors.
    """
    if not enabled:
        # If not enabled, return a tensor of the base epsilon value
        return torch.full_like(d_anchors, base_epsilon)

    # All calculations are now PyTorch tensor operations
    # Use the corrected argument names that match the config keys
    factor = 1.0 + kthr * (d_anchors - dthr)
    
    # Create a mask for where the adaptive logic should apply
    mask = d_anchors > dthr
    
    # Initialize epsilons with the base value
    eps = torch.full_like(d_anchors, base_epsilon)
    
    # Apply the adaptive factor only where the mask is True
    eps[mask] = base_epsilon * factor[mask]
    
    # Clamp the results to be within the min/max bounds
    return torch.clamp(eps, min=dmin, max=dmax)