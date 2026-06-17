"""
Condition preprocessing utilities.

On-the-fly Canny is the default for training (no precomputed condition files needed). Depth/pose
helpers are kept for extensibility.
"""

import cv2
import numpy as np


def make_canny(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """RGB uint8 image [H,W,3] -> 3-channel Canny edge map [H,W,3] (uint8)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.stack([edges] * 3, axis=-1)


def preprocess_canny(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """RGB/gray image -> single-channel Canny edges [H,W,1]."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges[..., None]


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a depth map to [0,1] and add a channel dim -> [H,W,1]."""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth_norm[..., None]
