# helper_functions.py
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tifffile
from PIL import Image
from skimage.morphology import skeletonize
from tensorflow.keras import backend as K
from patchify import patchify, unpatchify

# --- Path to trained model ---
MODEL_PATH = "models/12_viktoria_231781_unet_model_256px.h5"


# --- Utility Functions ---
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img)


def padder(
    image: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0
    )
    return padded, (top, bottom, left, right)


def unpadder(padded: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    h, w = padded.shape[:2]
    return padded[top : h - bottom, left : w - right]


def cropper(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    orig_shape = image.shape
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, {"original_shape": orig_shape, "used_crop": False}
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    xs, ys = max(cx - size // 2, 0), max(cy - size // 2, 0)
    cropped = image[ys : ys + size, xs : xs + size]
    return cropped, {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": xs,
        "y_start": ys,
        "crop_size": size,
    }


def uncropper(cropped: np.ndarray, info: Dict) -> np.ndarray:
    if not info.get("used_crop", False):
        return cropped
    h0, w0 = info["original_shape"]
    canvas = np.zeros((h0, w0), dtype=cropped.dtype)
    xs, ys, sz = info["x_start"], info["y_start"], info["crop_size"]
    canvas[ys : ys + sz, xs : xs + sz] = cropped
    return canvas


def threshold_mask(raw_mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    return (raw_mask > threshold).astype(np.uint8)


def morphological_closing(
    binary: np.ndarray,
    kernel_size=(3, 3),
    dilate_iter=5,
    erode_iter=3,
    kernel_shape=cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.erode(dilated, kernel, iterations=erode_iter)
    return closed.astype(np.float32)


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return TP / (Positives + K.epsilon())

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return TP / (Pred_Positives + K.epsilon())

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, custom_objects={"f1": f1})


def get_model():
    return load_model(MODEL_PATH)


def segment_image(
    model: tf.keras.Model, image_bytes: bytes, patch_size: int = 256, step: int = 128
) -> Dict[str, Any]:
    img = load_image_from_bytes(image_bytes)
    cropped, crop_info = cropper(img)
    padded, pad_info = padder(cropped, patch_size)
    patches = patchify(padded, (patch_size, patch_size), step=step)
    n_h, n_w, _, _ = patches.shape
    batch = patches.reshape(-1, patch_size, patch_size, 1).astype("float32") / 255.0
    preds = (
        model.predict(batch, verbose=0)
        .squeeze()
        .reshape(n_h, n_w, patch_size, patch_size)
    )
    full_padded = unpatchify(preds, padded.shape)
    unpadded = unpadder(full_padded, pad_info)
    mask = uncropper(unpadded, crop_info)
    return {
        "mask": mask.astype("float32"),
        "crop_info": crop_info,
        "pad_info": pad_info,
    }


def mask_to_tiff_bytes(mask_float: np.ndarray, metadata: Dict) -> bytes:
    mask_uint8 = (mask_float * 255).astype(np.uint8)
    buf = io.BytesIO()
    tifffile.imwrite(buf, mask_uint8, description=json.dumps(metadata))
    return buf.getvalue()


def save_mask_tif(image_path: str, mask: np.ndarray, metadata: Dict) -> str:
    base, _ = os.path.splitext(image_path)
    output_path = f"{base}_mask.tif"
    tifffile.imwrite(output_path, mask, description=json.dumps(metadata))
    return output_path
