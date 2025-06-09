# ─── Imports ──────────────────────────────────────────────────────────

import numpy as np
import cv2
import tensorflow as tf
from typing import Dict, Any, List
from patchify import patchify, unpatchify

from .io_utils import load_image_from_bytes
from .preprocessing_utils import cropper, padder, unpadder, uncropper

# ─── Segmentation functions ───────────────────────────────────────────

def segment_image(
    model: tf.keras.Model,
    image_bytes: bytes,
    patch_size: int = 256,
    step: int = 128
) -> Dict[str, Any]:
    """
    Decode raw bytes, run U-Net inference, and return the raw mask + metadata.

    Steps:
      1. load_image_from_bytes → 2D grayscale array
      2. cropper              → isolate Petri dish
      3. padder               → pad H/W to multiples of patch_size
      4. patchify + normalize → sliding‐window batch
      5. model.predict        → float32 predictions in [0,1]
      6. unpatchify           → reassemble full padded mask
      7. unpadder             → remove padding
      8. uncropper            → restore to original image frame

    Args:
      model:        a loaded tf.keras U-Net
      image_bytes:  raw bytes (from UploadFile.read())
      patch_size:   tile size used in training (default: 256)
      step:         sliding‐window stride (default: 128)

    Returns:
      {
        "mask":      2D float32 array (0–1) same shape as original image,
        "crop_info": metadata to invert the crop,
        "pad_info":  metadata to invert the padding
      }
    """
    # 1) Decode bytes → 2D uint8 grayscale
    img = load_image_from_bytes(image_bytes)

    # 2) Crop to Petri dish
    cropped, crop_info = cropper(img)

    # 3) Pad to multiples of patch_size
    padded, pad_info = padder(cropped, patch_size)

    # 4) Patchify & normalize
    patches = patchify(padded, (patch_size, patch_size), step=step)
    n_h, n_w, _, _ = patches.shape
    batch = (
        patches
        .reshape(-1, patch_size, patch_size, 1)
        .astype("float32") / 255.0
    )

    # 5) Predict on all tiles
    preds = model.predict(batch, verbose=0).squeeze()  # shape: (n_h*n_w, H, W)
    preds = preds.reshape(n_h, n_w, patch_size, patch_size)

    # 6) Reassemble padded mask
    full_padded = unpatchify(preds, padded.shape)

    # 7) Remove padding & restore crop
    unpadded = unpadder(full_padded, pad_info)
    mask = uncropper(unpadded, crop_info)

    return {
        "mask":      mask.astype("float32"),
        "crop_info": crop_info,
        "pad_info":  pad_info,
    }


def segment_plants_from_dish(mask: np.ndarray,
                              num_plants: int = 5,
                              min_area: int = 100,
                              aspect_ratio_threshold: float = 1.5,
                              vertical_start_thresh_ratio: float = 0.3,
                              angle_filter_area_thresh: int = 1200,
                              min_angle_degrees: float = 65,
                              edge_margin_px: int = 250
                             ) -> List[np.ndarray]:
    """
    Segment root regions from a cropped dish image into individual plant masks.

    Args:
        mask (np.ndarray): single-channel binary mask of the dish-cropped image.
        num_plants (int): expected number of plants (bands) across the width.
        min_area (int): minimum connected-component area to keep.
        aspect_ratio_threshold (float): min height/width ratio for vertical components.
        vertical_start_thresh_ratio (float): max y-start ratio to allow.
        angle_filter_area_thresh (int): area threshold to apply angle-based filter.
        min_angle_degrees (float): min angle for vertical component acceptance.
        edge_margin_px (int): margin from left/right edges to reject components.

    Returns:
        List[np.ndarray]: list of length num_plants, each a binary mask (uint8) of the largest root region.
    """
    h, w = mask.shape
    band_width = w / num_plants

    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # Prepare empty masks for each plant
    plant_masks: List[np.ndarray] = [np.zeros_like(mask, dtype=np.uint8) for _ in range(num_plants)]

    for i in range(1, num_labels):  # skip background label 0
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]

        # Basic area and shape filters
        if area < min_area:
            continue
        aspect_ratio = (ch / cw) if cw > 0 else 0
        if aspect_ratio < aspect_ratio_threshold:
            continue
        if y > int(h * vertical_start_thresh_ratio):
            continue

        # Additional angle-based filter for small components
        if area < angle_filter_area_thresh:
            angle = np.degrees(np.arctan2(ch, cw))
            too_small_angle = angle < min_angle_degrees
            too_close_left = x < edge_margin_px
            too_close_right = (x + cw) > (w - edge_margin_px)
            if too_small_angle or (too_close_left or too_close_right):
                continue

        # Determine plant band index
        band_idx = int(cx // band_width)
        band_idx = min(band_idx, num_plants - 1)

        # Create mask of this component
        component_mask = (labels == i).astype(np.uint8)
        # Keep the component if larger than any existing in this band
        if np.count_nonzero(component_mask) > np.count_nonzero(plant_masks[band_idx]):
            plant_masks[band_idx] = component_mask

    # Scale masks to 0/255 for consistency
    plant_masks = [(m * 255).astype(np.uint8) for m in plant_masks]
    return plant_masks


def merge_segmented_masks(segmented: List[np.ndarray],
                          crop_params: Dict[str, Any]
                         ) -> np.ndarray:
    """
    Merge the list of per-plant masks (all in the cropped frame) into one combined mask.
    
    Args:
        segmented: List of binary masks (0/255 uint8), each shape=(H_crop, W_crop).
        crop_params: dict with keys 'x_start','x_end','top_crop','orig_shape'.

    Returns:
        combined: single uint8 mask of shape (H_crop, W_crop) with all plants.
    """
    # They all share the same H_crop, W_crop
    H_crop = segmented[0].shape[0]
    W_crop = segmented[0].shape[1]
    
    combined = np.zeros((H_crop, W_crop), dtype=np.uint8)
    for m in segmented:
        # wherever any mask is white, turn that pixel white
        combined = np.maximum(combined, m)
    return combined


def reconstruct_full_mask(cropped_mask: np.ndarray,
                          crop_params: Dict[str, Any]
                         ) -> np.ndarray:
    """
    Pad a cropped mask back out to the original full-image size, using the stored params.
    
    Args:
        cropped_mask: 2D uint8 array of shape (H_crop, W_crop).
        crop_params: dict with:
            'x_start', 'x_end',    # columns in original
            'top_crop',            # rows removed from top
            'orig_shape'           # (H_full, W_full)
    Returns:
        full_mask: 2D uint8 array of shape orig_shape, with cropped_mask pasted in place.
    """
    H_full, W_full = crop_params['orig_shape']
    x_start  = crop_params['x_start']
    top_crop = crop_params['top_crop']
    # x_end not strictly needed but there if you want to double-check width
    
    full_mask = np.zeros((H_full, W_full), dtype=np.uint8)
    # paste the cropped_mask back
    full_mask[top_crop : top_crop + cropped_mask.shape[0],
              x_start : x_start  + cropped_mask.shape[1]
             ] = cropped_mask
    return full_mask
