# src/core/depth_image_library.py

import collections
from typing import List, Optional, Deque, Tuple, Any, Union

# Import both classes and create a Union type
from .depth_image import DepthImage as DepthImageLegacy
from .depth_image import DepthImage as DepthImageTorch

# This tells the type checker that the library can hold either version.
DepthImageTypes = Union[DepthImageLegacy, DepthImageTorch]

class DepthImageLibrary:
    """
    Manages a library of DepthImage objects, typically with a fixed maximum size.
    This version is generic and can hold either legacy or torch-based DepthImage objects.
    """
    def __init__(self, max_size: int):
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        self.max_size: int = max_size
        # The deque will hold objects of type DepthImageTypes
        self._images: Deque[DepthImageTypes] = collections.deque(maxlen=max_size)

    def add_image(self, depth_image: Any) -> None: # Changed hint to Any for flexibility
        """
        Adds a new DepthImage to the library.
        """
        # The check is now more flexible. It ensures the object is one of our two types.
        # This allows the same library code to be used by both systems.
        if not isinstance(depth_image, (DepthImageLegacy, DepthImageTorch)):
            raise TypeError(f"Only DepthImageLegacy or DepthImageTorch objects can be added. Got {type(depth_image)}.")
        self._images.append(depth_image)

    def get_image_by_index(self, index: int) -> Optional[DepthImageTypes]:
        """
        Retrieve a DepthImage by its index in the internal deque.
        """
        if not self._images:
            return None
        try:
            return self._images[index]
        except IndexError:
            return None

    # --- Methods that need to be careful about the object type ---
    
    def get_relevant_past_images(self, num_sweeps: int) -> List[Tuple[int, Any]]:
        """
        Retrieves a specified number of the most recent past images from the library.
        Returns a list of (index, DepthImage) tuples.
        """
        if not self._images or num_sweeps <= 0:
            return []
        
        num_available = len(self._images)
        num_to_get = min(num_sweeps, num_available)
        
        relevant_dis = []
        for i in range(num_to_get):
            # Index from the end: -1 is the last item, -2 is the second to last, etc.
            image_index_from_end = -1 - i
            actual_index_in_deque = num_available + image_index_from_end
            relevant_dis.append((actual_index_in_deque, self._images[image_index_from_end]))
            
        # The list is already sorted from most recent to least recent.
        return relevant_dis


    # ... other methods like get_relevant_future_images, __len__, etc., remain the same ...
    # They operate on the deque and don't need to know the internal details of the DI objects.
    # The provided code for the rest of the file is fine.
    def get_relevant_future_images(self, current_timestamp: float, time_window_s: float) -> List[Tuple[int, DepthImageTypes]]:
        """
        Retrieves future images within a specified time window from the current_timestamp.
        Returns a list of (original_index_in_deque, DepthImage) tuples, sorted by
        timestamp closest to current_timestamp first (i.e., nearest future images first).
        """
        relevant_dis: List[Tuple[int, DepthImageTypes]] = []
        for i, di_candidate in enumerate(self._images):
            if di_candidate.timestamp > current_timestamp and \
               (di_candidate.timestamp - current_timestamp) <= time_window_s * 1e6:
                relevant_dis.append((i, di_candidate))
        
        relevant_dis.sort(key=lambda x: x[1].timestamp - current_timestamp)
        return relevant_dis

    def __len__(self) -> int:
        return len(self._images)

    def get_all_images(self) -> List[DepthImageTypes]:
        return list(self._images)

    def is_full(self) -> bool:
        return len(self._images) == self.max_size

    def clear(self) -> None:
        self._images.clear()

    def __str__(self) -> str:
        return f"DepthImageLibrary(size={len(self._images)}, max_size={self.max_size})"