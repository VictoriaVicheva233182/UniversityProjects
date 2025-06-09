# ─── Imports ──────────────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
import io
import numpy as np
from PIL import Image
from pathlib import Path

from app.utils.io_utils import load_image_from_bytes, mask_to_tiff_bytes, save_mask_tif
from app.utils.model_utils import load_model
from app.utils.segmentation_utils import segment_image
from app.utils.postprocessing_utils import threshold_mask, morphological_closing

# ─── FastAPI setup ────────────────────────────────────────────────────

router = APIRouter(prefix="/segment", tags=["Segmentation"])


# Get the absolute path to the folder this script is in
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Define model and output paths based on the current file
MODEL_PATH = BASE_DIR / "models" / "12_viktoria_231781_unet_model_256px.h5"
OUTPUT_DIR = BASE_DIR / "masks_dir"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load the trained model once
model = load_model(str(MODEL_PATH))

# ─── API endpoint ─────────────────────────────────────────────────────

@router.post("/", summary="Segment an image into a binary mask")

async def segment_endpoint(
    file: UploadFile = File(...),
    threshold: float = Query(0.1, description="Binarization threshold")
):
    # 1) Validate & read
    if file.content_type not in ("image/png","image/jpeg","image/tiff","image/tif"):
        raise HTTPException(415, "Unsupported file type")
    image_bytes = await file.read()

    try:
        # 2) Segmentation pipeline
        seg_result = segment_image(model, image_bytes)
        raw_mask   = seg_result["mask"]  # float32 [0–1]

        # 3) Binarize & refine
        binary_mask = threshold_mask(raw_mask, threshold)
        closed_mask = morphological_closing(binary_mask)

        # 4) Prepare metadata
        metadata = {
            "crop_info": seg_result.get("crop_info"),
            "pad_info":  seg_result.get("pad_info")
        }

        # 5) Encode TIFF bytes & save to disk
        tiff_uint8 = (closed_mask * 255).astype(np.uint8)
        tiff_bytes = mask_to_tiff_bytes(tiff_uint8, metadata)
        base       = Path(file.filename).stem
        out_path = OUTPUT_DIR / f"{base}_mask.tif"
        save_mask_tif(str(out_path), tiff_uint8, metadata)

        # 6) Build PNG preview
        png_img = Image.fromarray(tiff_uint8, mode="L")
        png_buf = io.BytesIO()
        png_img.save(png_buf, format="PNG")
        png_buf.seek(0)

        # 7) Return PNG for inline display in Swagger/UI
        return StreamingResponse(
            png_buf,
            media_type="image/png",
            headers={"Content-Disposition":f"inline; filename={base}_mask_preview.png"}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
