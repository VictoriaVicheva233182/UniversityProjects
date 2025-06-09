# ─── Imports ──────────────────────────────────────────────────────────

import cv2
import numpy as np
from typing import Tuple, Dict, Any

# ─── Postprocessing functions ─────────────────────────────────────────

def threshold_mask(raw_mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Convert a raw float mask (0-1) to a binary mask using given threshold.
    """
    return (raw_mask > threshold).astype(np.uint8)


def morphological_closing(binary: np.ndarray,
               kernel_size: tuple = (3,3),
               dilate_iter: int = 5,
               erode_iter: int = 3,
               kernel_shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion) to refine binary mask.
    """
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.erode(dilated, kernel, iterations=erode_iter)
    
    # Convert 0/1 uint8 → 0.0/1.0 float32
    return closed.astype(np.float32)


def crop_top_and_dish(binary: np.ndarray,
                      crop_info: Dict[str, Any],
                      top_crop_ratio: float = 0.15
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Crop a single plant mask to Petri dish bounds and remove the top portion.

    Args:
        binary (np.ndarray): single-plant binary mask.
        crop_info (dict): contains 'x_start' and 'crop_size' for horizontal crop.
        top_crop_ratio (float): fraction of mask height to crop from top.

    Returns:
        cropped (np.ndarray): the cropped binary mask.
        params (dict): parameters needed to reconstruct the original mask:
            - x_start: left index of horizontal crop
            - x_end: right index of horizontal crop
            - top_crop: number of rows removed from top
            - orig_shape: original mask shape (height, width)
    """
    x_start = crop_info['x_start']
    crop_size = crop_info['crop_size']
    x_end = min(x_start + crop_size, binary.shape[1])

    # Horizontal crop to dish width
    dish = binary[:, x_start:x_end]

    # Vertical crop to remove top noise region
    top_crop = int(dish.shape[0] * top_crop_ratio)
    cropped = dish[top_crop:, :]

    # Store reconstruction parameters
    params = {
        'x_start': x_start,
        'x_end': x_end,
        'top_crop': top_crop,
        'orig_shape': binary.shape
    }

    return cropped, params