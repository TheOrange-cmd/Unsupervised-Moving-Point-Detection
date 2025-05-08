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

    # # potentially unused now
    # def get_latest_image(self) -> Optional[DepthImage]:
    #     """
    #     Retrieves the most recently added (newest) DepthImage from the library.

    #     Returns:
    #         Optional[DepthImage]: The latest DepthImage, or None if the library is empty.
    #     """
    #     if not self._images:
    #         return None
    #     return self._images[-1] # The rightmost element in deque is the newest
    
    
    # def get_nearest_frames_by_timestamp(self, target_timestamp: float, past_count: int, future_count: int) -> Optional[Dict[str, List[DepthImage]]]:
    #     """
    #     Finds a 'center' frame closest to the target_timestamp and returns a temporal
    #     context of `past_count` frames before it and `future_count` frames after it.

    #     Args:
    #         target_timestamp (float): The timestamp to find the closest 'center' frame to.
    #         past_count (int): Number of past frames to retrieve, relative to the center frame.
    #         future_count (int): Number of future frames to retrieve, relative to the center frame.

    #     Returns:
    #         Optional[Dict[str, List[DepthImage]]]: A dictionary with keys 'past', 'center', 'future'.
    #             'center' is a list containing the single closest frame.
    #             'past' and 'future' are lists of DepthImages.
    #             Returns None if no images are in the library or a center frame cannot be determined.
    #     """
    #     if not self._images:
    #         return None
            
    #     center_di_candidate: Optional[DepthImage] = None
    #     min_diff = float('inf')
    #     center_idx: int = -1

    #     for i, img in enumerate(self._images):
    #         diff = abs(img.timestamp - target_timestamp)
    #         if diff < min_diff:
    #             min_diff = diff
    #             center_di_candidate = img
    #             center_idx = i
    #         elif diff == min_diff and center_di_candidate and img.timestamp > center_di_candidate.timestamp:
    #             # Prefer later frame if timestamps are equally different (arbitrary tie-break)
    #             center_di_candidate = img
    #             center_idx = i
                
    #     if center_di_candidate is None: # Should not happen if self._images is not empty
    #         return None

    #     # Get temporal context around this frame
        
    #     past_images: List[DepthImage] = []
    #     if past_count > 0 and center_idx > 0:
    #         start_idx = max(0, center_idx - past_count)
    #         past_images = list(self._images)[start_idx:center_idx]
    #         past_images.reverse() # Typically want most recent past first

    #     future_images: List[DepthImage] = []
    #     if future_count > 0 and center_idx < len(self._images) - 1:
    #         end_idx = min(len(self._images), center_idx + 1 + future_count)
    #         future_images = list(self._images)[center_idx + 1:end_idx]
            
    #     return {
    #         'center': [center_di_candidate], # Return as a list for consistency
    #         'past': past_images,
    #         'future': future_images
    #     }

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