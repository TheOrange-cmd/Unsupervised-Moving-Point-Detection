# src/core/m_detector/adaptive_epsilon_utils.py

import torch

def calculate_adaptive_epsilon(
    d_anchors: torch.Tensor,
    base_epsilon: float,
    enabled: bool,
    d_threshold: float,
    k_threshold: float,
    d_max: float,
    d_min: float
) -> torch.Tensor:
    """
    Calculates an adaptive epsilon value for a batch of points based on their anchor depths.
    This is a pure PyTorch implementation.

    Args:
        d_anchors (torch.Tensor): A 1D tensor of anchor depths for each point.
        base_epsilon (float): The base epsilon value.
        enabled (bool): Whether the adaptive logic is enabled.
        d_threshold (float): Depth threshold for applying the linear scaling factor.
        k_threshold (float): Scaling factor for depths beyond the threshold.
        d_max (float): The maximum possible epsilon value.
        d_min (float): The minimum possible epsilon value.

    Returns:
        torch.Tensor: A 1D tensor of calculated epsilon values, same size as d_anchors.
    """
    if not enabled:
        # If not enabled, return a tensor of the base epsilon value
        return torch.full_like(d_anchors, base_epsilon)

    # All calculations are now PyTorch tensor operations
    factor = 1.0 + k_threshold * (d_anchors - d_threshold)
    
    # Create a mask for where the adaptive logic should apply
    # We use torch.clamp to prevent d_anchors from being None or non-tensor
    mask = d_anchors > d_threshold
    
    # Initialize epsilons with the base value
    eps = torch.full_like(d_anchors, base_epsilon)
    
    # Apply the adaptive factor only where the mask is True
    eps[mask] = base_epsilon * factor[mask]
    
    # Clamp the results to be within the min/max bounds
    # torch.clamp is the equivalent of np.clip
    return torch.clamp(eps, min=d_min, max=d_max)