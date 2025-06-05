# app/postprocessing.py

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from PIL import Image
import io

__all__ = [
    "threshold_mask",
    "morphological_closing",
    "crop_to_dish",
    "crop_top_only",
    "crop_top_and_dish",
    "segment_roots",
    "mask_to_png_bytes",
]


def threshold_mask(raw_mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    return (raw_mask > threshold).astype(np.uint8)


def morphological_closing(
    binary: np.ndarray,
    kernel_size: tuple = (3, 3),
    dilate_iter: int = 5,
    erode_iter: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.erode(dilated, kernel, iterations=erode_iter)
    return closed


def crop_to_dish(closed_mask: np.ndarray, crop_info: Dict[str, Any]) -> np.ndarray:
    xs = crop_info["x_start"]
    ys = crop_info["y_start"]
    size = crop_info["crop_size"]
    return closed_mask[ys : ys + size, xs : xs + size]


def crop_top_only(
    mask: np.ndarray, top_crop_ratio: float = 0.15
) -> Tuple[np.ndarray, int]:
    h = mask.shape[0]
    y_off = int(h * top_crop_ratio)
    return mask[y_off:, :], y_off


def crop_top_and_dish(
    binary: np.ndarray, crop_info: Dict[str, int], top_crop_ratio: float = 0.15
) -> Tuple[np.ndarray, int]:
    x0 = crop_info["x_start"]
    size = crop_info["crop_size"]
    dish = binary[:, x0 : x0 + size]
    y_off = int(dish.shape[0] * top_crop_ratio)
    return dish[y_off:], y_off


def segment_roots(
    dish_mask: np.ndarray,
    num_plants: int = 5,
    min_area: int = 30,  # ↓ relaxed
    aspect_ratio_thresh: float = 1.2,  # ↓ relaxed
    vertical_start_thresh: float = 0.5,  # ↑ allows lower roots
    angle_area_thresh: int = 800,  # ↓ relaxed
    min_angle: float = 30.0,  # ↓ allows shallow roots
    edge_margin: int = 100,  # ↓ allows edge roots
) -> List[np.ndarray]:
    h, w = dish_mask.shape
    bands = w / num_plants
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dish_mask, connectivity=8
    )
    plant_masks = [np.zeros_like(dish_mask, dtype=np.uint8) for _ in range(num_plants)]

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]

        if area < min_area:
            continue

        ar = ch / cw if cw > 0 else 0
        if ar < aspect_ratio_thresh:
            continue

        if y > h * vertical_start_thresh:
            continue

        if area < angle_area_thresh:
            angle = np.degrees(np.arctan2(ch, cw))
            if angle < min_angle:
                continue

        if x < edge_margin or (x + cw) > (w - edge_margin):
            continue

        band = int(cx // bands)
        band = min(band, num_plants - 1)
        comp = (labels == i).astype(np.uint8)

        if comp.sum() > plant_masks[band].sum():
            plant_masks[band] = comp

    return plant_masks


def mask_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode a BGR or grayscale uint8 image to PNG bytes."""
    mode = "RGB" if img.ndim == 3 and img.shape[2] == 3 else "L"
    pil = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
