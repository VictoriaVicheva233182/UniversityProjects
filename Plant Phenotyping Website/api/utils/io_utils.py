# ─── Imports ─────────────────────────────────────────────────────────

import io
from typing import Dict, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import json

# ─── Input functions ───────────────────────────────────────────────── 

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load raw bytes into a H×W uint8 grayscale array.
    (This preserves the original resolution so you can tile it.)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img)


def load_mask_tif(
    mask_src: Union[str, io.BytesIO]
) -> Tuple[np.ndarray, Dict]:
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        desc = tif.pages[0].tags["ImageDescription"].value
    metadata = json.loads(desc) if desc else {}
    return mask, metadata

# ─── Output functions ─────────────────────────────────────────────────

def mask_to_tiff_bytes(
    mask_float: np.ndarray,
    metadata: Dict
) -> bytes:
    """
    Encode a float‐mask + metadata into an in‐memory TIFF byte string.

    Steps:
      1. Scale float mask [0–1] to uint8 [0–255]
      2. Write to a BytesIO with tifffile, embedding metadata as JSON in ImageDescription

    Args:
      mask_float: 2D float32 numpy array with values in [0,1].
      metadata:   Dict to JSON‐serialize into the TIFF ImageDescription tag.

    Returns:
      Raw bytes of the resulting TIFF.
    """
    # 1) Convert to uint8
    mask_uint8 = (mask_float * 255).astype(np.uint8)

    # 2) Write to in-memory TIFF
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        mask_uint8,
        description=json.dumps(metadata)
    )
    return buf.getvalue()


def save_mask_tif(image_path: str, mask_uint8: np.ndarray, metadata: Dict) -> None:
    tifffile.imwrite(image_path, mask_uint8, description=json.dumps(metadata))