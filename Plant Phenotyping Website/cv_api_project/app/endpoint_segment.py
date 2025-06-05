from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
import numpy as np
import io
import os
from PIL import Image
import tensorflow as tf
from typing import Dict, Any

from app.helper_functions import (
    threshold_mask,
    morphological_closing,
    f1,
    mask_to_tiff_bytes,
    segment_image,
)

router = APIRouter()

# Configuration
MODEL_PATH = "models/12_viktoria_231781_unet_model_256px.h5"
OUTPUT_DIR = Path("debug_inputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the U-Net model at startup
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"f1": f1})


@router.post("/segment")
async def segment_endpoint(file: UploadFile = File(...), threshold: float = 0.1):
    """
    Upload an image, run segmentation, and return a binary mask TIFF with metadata.
    """
    if file.content_type not in ["image/png", "image/jpeg", "image/tiff", "image/tif"]:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        image_bytes = await file.read()
        seg_result = segment_image(model, image_bytes)
        raw_mask = seg_result["mask"]

        binary_mask = threshold_mask(raw_mask, threshold)
        closed_mask = morphological_closing(binary_mask)

        metadata = {
            "crop_info": seg_result.get("crop_info"),
            "pad_info": seg_result.get("pad_info"),
        }

        tiff_bytes = mask_to_tiff_bytes(closed_mask, metadata)

        base_name = Path(file.filename).stem
        output_path = OUTPUT_DIR / f"{base_name}_mask.tif"
        with open(output_path, "wb") as out_file:
            out_file.write(tiff_bytes)

        return StreamingResponse(
            io.BytesIO(tiff_bytes),
            media_type="image/tiff",
            headers={"Content-Disposition": f"inline; filename={output_path.name}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
