# ─── Imports ──────────────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import numpy as np
from PIL import Image
import os

from app.utils.model_utils import load_model
from app.utils.segmentation_utils import segment_image
from app.utils.postprocessing_utils import threshold_mask, morphological_closing, crop_top_and_dish
from app.utils.segmentation_utils import segment_plants_from_dish
from app.utils.analysis_utils import measure_primary_root_and_tip, adjust_measurements_to_full, overlay_roots_on_image

# ─── FastAPI setup ────────────────────────────────────────────────────

router = APIRouter(prefix="/analyze_image", tags=["Image Analysis"])

# Configuration
from pathlib import Path

# Always to resolve based on this file's location
# Configuration

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "12_viktoria_231781_unet_model_256px.h5"
OUTPUT_DIR = BASE_DIR / "images_labelled_dir"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the trained model once
model = load_model(str(MODEL_PATH))

# ─── API endpoint ─────────────────────────────────────────────────────

@router.post("/", summary="Analyze roots and overlay measurements on image")

async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.1, description="Mask binarization threshold")
):
    # Validate upload
    if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/tif"):
        raise HTTPException(415, "Unsupported file type")
    image_bytes = await file.read()

    # 1) Full-image segmentation
    seg = segment_image(model, image_bytes)
    raw_mask = seg["mask"]

    # 2) Binarize and refine
    binary = threshold_mask(raw_mask, threshold)
    closed = morphological_closing(binary)
    closed_u8 = (closed * 255).astype(np.uint8)

    # 3) Crop to dish and split into individual plants
    crop_info = seg.get("crop_info")
    if not crop_info:
        raise HTTPException(400, "TIFF metadata missing 'crop_info'")
    cropped, crop_params = crop_top_and_dish(closed_u8, crop_info)
    segments = segment_plants_from_dish(cropped)

    # 4) Measure primary roots
    root_lengths, root_tips, root_paths = {}, {}, {}
    for i, pmask in enumerate(segments, start=1):
        key = f"Plant_{i}"
        length, tip, path = measure_primary_root_and_tip(pmask)
        root_lengths[key] = length
        root_tips[key] = tip
        root_paths[key] = path

    # 5) Adjust coordinates to full-frame
    tips_full, paths_full = adjust_measurements_to_full(root_tips, root_paths, crop_params)

    # 6) Overlay on original image
    overlay_img, measurement_strs = overlay_roots_on_image(
        image_bytes, tips_full, paths_full, root_lengths
    )

     # ── print them to your console ──
    print("Measurements:")
    for plant, desc in measurement_strs.items():
        print(f"   • {plant}: {desc}")

    # 7) Save overlay PNG
    base = Path(file.filename).stem
    out_png = OUTPUT_DIR / f"{base}_overlay.png"
    overlay_img.save(out_png, format="PNG")

    # 8) Return inline PNG preview
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)

    # 8) Return inline PNG
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={base}_overlay.png"}
    )
