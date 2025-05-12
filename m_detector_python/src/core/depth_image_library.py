"""
Manages a collection of DepthImage objects, typically with a fixed maximum size,
acting as a sliding window of recent depth image data.
"""

# src/core/depth_image_library.py

import collections
from typing import List, Optional, Deque
from .depth_image import DepthImage 

class DepthImageLibrary:
    """
    Manages a library of DepthImage objects, typically with a fixed maximum size.
    New images are added, and if the library exceeds its maximum size, the oldest
    image is discarded.
    """
    def __init__(self, max_size: int):
        """
        Initializes the DepthImageLibrary.

        Args:
            max_size (int): The maximum number of DepthImage objects to store.
                            Must be a positive integer.
        """
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        self.max_size: int = max_size
        self._images: Deque[DepthImage] = collections.deque(maxlen=max_size)

    def add_image(self, depth_image: DepthImage) -> None:
        """
        Adds a new DepthImage to the library.
        If the library is already at max_size, the oldest image is automatically
        removed due to the deque's maxlen property.

        Args:
            depth_image (DepthImage): The DepthImage object to add.
        """
        if not isinstance(depth_image, DepthImage):
            raise TypeError("Only DepthImage objects can be added to the library.")
        self._images.append(depth_image)

    def get_image_by_index(self, index: int) -> Optional[DepthImage]:
        """
        Retrieve a DepthImage by its index in the internal deque.
        Index 0 is the oldest image, -1 is the newest.

        Args:
            index (int): Index of the DepthImage to retrieve.

        Returns:
            Optional[DepthImage]: The requested DepthImage, or None if index is invalid
                                  or the library is empty.
        """
        if not self._images:
            return None
        try:
            return self._images[index]
        except IndexError:
            return None


    def get_image_by_timestamp(self, timestamp: float, mode: str = 'closest') -> Optional[DepthImage]:
        """
        Finds a DepthImage based on the provided timestamp and search mode.

        Args:
            timestamp (float): The target timestamp to search around.
            mode (str): Search mode. Can be 'closest', 'before' (closest image with
                        timestamp <= target), or 'after' (closest image with
                        timestamp >= target). Defaults to 'closest'.

        Returns:
            Optional[DepthImage]: The found DepthImage, or None if no suitable image
                                  is found according to the mode.
        """
        if not self._images:
            return None

        closest_image: Optional[DepthImage] = None
        min_time_diff = float('inf')
        
        # For 'before' and 'after', we might need to find the best candidate
        # that satisfies the condition, not just the absolute closest.
        candidate_images: List[DepthImage] = []

        if mode == 'closest':
            candidate_images = list(self._images)
        elif mode == 'before':
            candidate_images = [img for img in self._images if img.timestamp <= timestamp]
        elif mode == 'after':
            candidate_images = [img for img in self._images if img.timestamp >= timestamp]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'closest', 'before', or 'after'.")

        if not candidate_images:
            return None

        for image in candidate_images:
            time_diff = abs(image.timestamp - timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_image = image
            # Special handling for 'before' and 'after' to get the *closest* that satisfies the boundary
            elif time_diff == min_time_diff:
                if mode == 'before' and image.timestamp > (closest_image.timestamp if closest_image else -float('inf')):
                    closest_image = image # Prefer later timestamp if equally close for 'before'
                elif mode == 'after' and image.timestamp < (closest_image.timestamp if closest_image else float('inf')):
                    closest_image = image # Prefer earlier timestamp if equally close for 'after'
                    
        return closest_image

    def is_ready_for_processing(self, min_images: int) -> bool:
        """
        Checks if the library contains at least a minimum number of images.

        Args:
            min_images (int): The minimum number of images required.

        Returns:
            bool: True if the library has `min_images` or more, False otherwise.
        """
        return len(self._images) >= min_images


    def __len__(self) -> int:
        """Return the number of DepthImages currently in the library."""
        return len(self._images)

    def get_all_images(self) -> List[DepthImage]:
        """
        Retrieves all DepthImage objects currently in the library,
        ordered from oldest to newest.

        Returns:
            List[DepthImage]: A list of all DepthImage objects.
        """
        return list(self._images)

    def is_full(self) -> bool:
        """
        Checks if the library has reached its maximum capacity.
        """
        return len(self._images) == self.max_size

    def clear(self) -> None:
        """
        Removes all images from the library.
        """
        self._images.clear()

    def __str__(self) -> str:
        return f"DepthImageLibrary(size={len(self._images)}, max_size={self.max_size})"