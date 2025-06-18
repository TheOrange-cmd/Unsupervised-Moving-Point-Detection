# src/utils/seeding_utils.py

import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    """
    Sets the seed for all relevant random number generators to ensure reproducibility.
    
    Args:
        seed (int): The integer value to use as the seed.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch on GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # The two lines below are often needed for full reproducibility with CUDA
        # They can result in a performance hit, but are essential for deterministic behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Set a hash seed for python for certain hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)