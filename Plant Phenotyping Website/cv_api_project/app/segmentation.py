# plantphenotyping/segmentation.py
import io
import cv2
import numpy as np
import tensorflow as tf
import networkx as nx
from typing import List, Dict, Tuple
from skimage.morphology import skeletonize
from patchify import patchify, unpatchify
from PIL import Image
from tensorflow.keras import backend as K
from typing import Any, Dict


from .preprocessing import (
    load_image_from_bytes,
    cropper,
    padder,
    unpadder,
    uncropper,
)

__all__ = [
    "f1",
    "load_model",
    "segment_image"
]

#  Custom F1 metric for model loading
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_model(model_path: str) -> tf.keras.Model:
    """Load a U‑Net/ResNet model with custom F1 metric."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"f1": f1}
    )


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
