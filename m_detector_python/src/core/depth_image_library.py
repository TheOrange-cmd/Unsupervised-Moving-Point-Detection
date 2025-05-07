# src/core/depth_image_library.py

import collections
from typing import List, Optional, Deque
from .depth_image import DepthImage # Assuming DepthImage is in src/core/depth_image.py

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

    def get_latest_image(self) -> Optional[DepthImage]:
        """
        Retrieves the most recently added DepthImage.

        Returns:
            Optional[DepthImage]: The latest DepthImage, or None if the library is empty.
        """
        if not self._images:
            return None
        return self._images[-1] # The rightmost element is the newest

    def get_all_images(self) -> List[DepthImage]:
        """
        Retrieves all DepthImage objects currently in the library,
        ordered from oldest to newest.

        Returns:
            List[DepthImage]: A list of all DepthImage objects.
        """
        return list(self._images)

    def get_image_by_index(self, index: int) -> Optional[DepthImage]:
        """
        Retrieves an image by its index.
        Index 0 is the oldest, -1 is the newest (if library is not empty).
        Supports standard list-like indexing.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            Optional[DepthImage]: The DepthImage at the given index, or None if
                                 the index is out of bounds or library is empty.
        """
        try:
            return self._images[index]
        except IndexError:
            return None

    def __len__(self) -> int:
        """
        Returns the current number of images in the library.
        """
        return len(self._images)

    def is_full(self) -> bool:
        """
        Checks if the library has reached its maximum capacity.
        """
        return len(self._images) == self.max_size

    def is_ready_for_processing(self, min_images_required: int) -> bool:
        """
        Checks if the library has collected enough images to start a certain process
        (e.g., the M-detector's main loop after an initialization phase).

        Args:
            min_images_required (int): The minimum number of images needed.

        Returns:
            bool: True if the library has at least min_images_required images, False otherwise.
        """
        if min_images_required <= 0:
            return True # No minimum requirement means it's always ready
        return len(self._images) >= min_images_required

    def clear(self) -> None:
        """
        Removes all images from the library.
        """
        self._images.clear()

    def __str__(self) -> str:
        return f"DepthImageLibrary(size={len(self._images)}, max_size={self.max_size})"