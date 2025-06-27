# src/core/depth_image_library.py

import collections
from typing import List, Optional, Deque, Tuple, Any

from .depth_image import DepthImage

class DepthImageLibrary:
    """
    Manages a rolling library of DepthImage objects with a fixed maximum size.

    This class acts as the short-term memory for the M-Detector, holding a deque
    of the most recent LiDAR sweeps, each encapsulated in a DepthImage object.
    """
    def __init__(self, max_size: int):
        """
        Initializes the library.

        Args:
            max_size (int): The maximum number of DepthImage objects to store.
                            Once the deque is full, older images are discarded.
        """
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        self.max_size: int = max_size
        self._images: Deque[DepthImage] = collections.deque(maxlen=max_size)

    def add_image(self, depth_image: DepthImage) -> None:
        """Adds a new DepthImage to the end of the library."""
        if not isinstance(depth_image, DepthImage):
            raise TypeError(f"Only DepthImage objects can be added. Got {type(depth_image)}.")
        self._images.append(depth_image)

    def get_image_by_index(self, index: int) -> Optional[DepthImage]:
        """
        Retrieves a DepthImage by its index in the internal deque.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            Optional[DepthImage]: The DepthImage object at the given index, or None
                                  if the index is out of bounds.
        """
        if not self._images:
            return None
        try:
            return self._images[index]
        except IndexError:
            return None

    def get_relevant_past_images(self, num_sweeps: int) -> List[Tuple[int, DepthImage]]:
        """
        Retrieves a specified number of the most recent past images, which are
        used as the historical reference for occlusion checks.

        Args:
            num_sweeps (int): The number of recent past images to retrieve.

        Returns:
            List[Tuple[int, DepthImage]]: A list of (index, DepthImage) tuples,
                                          sorted from most recent to least recent.
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
            
        return relevant_dis

    def get_relevant_future_images(self, num_sweeps: int) -> List[Tuple[int, DepthImage]]:
        """
        Retrieves a specified number of the nearest future images from the library.
        This is currently unused but provides utility for planned bidirectional processing.

        Args:
            num_sweeps (int): The number of future images to retrieve relative to the
                              end of the deque.

        Returns:
            List[Tuple[int, DepthImage]]: A list of (index, DepthImage) tuples for future
                                          images, sorted from nearest future to furthest.
        """
        # This implementation assumes the "current" frame is at index -num_sweeps-1
        # and we are looking at frames after it. 
        if not self._images or num_sweeps <= 0 or len(self._images) <= num_sweeps:
            return []

        num_available = len(self._images)
        relevant_dis = []
        
        # Iterate from the sweep just after the "current" one to the end.
        for i in range(num_sweeps):
            # Index from the end: -1 is the last, -2 is second to last...
            image_index_from_end = -1 - i
            actual_index_in_deque = num_available + image_index_from_end
            relevant_dis.append((actual_index_in_deque, self._images[image_index_from_end]))
        
        # The list is already sorted from furthest future to nearest. Reverse it.
        return relevant_dis[::-1]

    def __len__(self) -> int:
        """Returns the current number of images in the library."""
        return len(self._images)

    def get_all_images(self) -> List[DepthImage]:
        """Returns a list of all images currently in the library."""
        return list(self._images)

    def is_full(self) -> bool:
        """Checks if the library has reached its maximum capacity."""
        return len(self._images) == self.max_size

    def clear(self) -> None:
        """Removes all images from the library."""
        self._images.clear()

    def __str__(self) -> str:
        return f"DepthImageLibrary(size={len(self._images)}, max_size={self.max_size})"