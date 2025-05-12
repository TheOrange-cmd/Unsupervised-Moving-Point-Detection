# src/visualization/visualization_utils.py
import numpy as np

import cv2 # For mpl_fig_to_opencv_bgr


def mpl_fig_to_opencv_bgr(fig):
    """Converts a Matplotlib figure to an OpenCV BGR image."""
    fig.canvas.draw()
    img_np_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_np_rgb = img_np_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    return img_np_bgr
