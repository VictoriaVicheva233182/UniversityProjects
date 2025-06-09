# ─── Imports ─────────────────────────────────────────────────────────

import cv2
import numpy as np
from typing import Tuple, Dict

# ─── Preprocessing functions ─────────────────────────────────────────

def padder(image: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Pads `image` so H/W are divisible by `patch_size`.
    Returns (padded_image, (top, bottom, left, right)).
    """
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding  = ((w // patch_size) + 1) * patch_size - w

    top    = height_padding // 2
    bottom = height_padding - top
    left   = width_padding  // 2
    right  = width_padding  - left

    padded = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return padded, (top, bottom, left, right)


def unpadder(padded: np.ndarray, pads: Tuple[int,int,int,int]) -> np.ndarray:
    """
    Remove previously added padding from `padded` using `pads` = (top, bottom, left, right).
    """
    top, bottom, left, right = pads
    h, w = padded.shape[:2]
    return padded[top : h-bottom, left : w-right]


def cropper(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Detect largest external contour (Petri dish), crop to a square around it,
    and return (cropped, crop_info). If none found, returns original + flag.
    """
    orig_shape = image.shape
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, {"original_shape": orig_shape, "used_crop": False}

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h)
    cx, cy = x + w//2, y + h//2
    xs = max(cx - size//2, 0)
    ys = max(cy - size//2, 0)

    cropped = image[ys:ys+size, xs:xs+size]
    return cropped, {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": xs,
        "y_start": ys,
        "crop_size": size
    }


def uncropper(cropped: np.ndarray, info: Dict) -> np.ndarray:
    """
    Place `cropped` back into a blank canvas of its `original_shape`,
    at (x_start, y_start) from `info`.
    """
    if not info.get("used_crop", False):
        return cropped

    h0, w0 = info["original_shape"]
    canvas = np.zeros((h0, w0), dtype=cropped.dtype)
    xs, ys, sz = info["x_start"], info["y_start"], info["crop_size"]
    canvas[ys:ys+sz, xs:xs+sz] = cropped
    return canvas

