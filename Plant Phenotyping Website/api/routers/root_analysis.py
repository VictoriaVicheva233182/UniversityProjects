# ─── Imports ──────────────────────────────────────────────────────────

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
import io

from app.utils.io_utils import load_mask_tif
from app.utils.postprocessing_utils import crop_top_and_dish
from app.utils.segmentation_utils import (
    segment_plants_from_dish,
    merge_segmented_masks,
    reconstruct_full_mask
)
from app.utils.analysis_utils import (
    measure_primary_root_and_tip,
    adjust_measurements_to_full,
    render_full_mask_with_roots_tiff
)
from PIL import Image
from pathlib import Path

# ─── FastAPI setup ────────────────────────────────────────────────────

router = APIRouter(prefix="/analyze_mask", tags=["Mask Analysis"])


# Get the absolute path to the current file's directory
BASE_DIR = Path(__file__).resolve().parent

# Define the output directory path
OUTPUT_DIR = BASE_DIR / "masks_labelled_dir"

# Create the directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── API endpoint ─────────────────────────────────────────────────────

@router.post("/", summary="Process a mask TIFF and overlay root measurements")

async def process_mask(file: UploadFile = File(...)):
    try:
        # 1) Read upload into memory
        data = await file.read()
        buf = io.BytesIO(data)

        # 2) Load mask + crop metadata
        mask_full, full_meta = load_mask_tif(buf)
        crop_info = full_meta.get("crop_info")
        pad_info = full_meta.get("pad_info")
        if crop_info is None:
            raise HTTPException(400, "TIFF is missing 'crop_info' metadata")

        # 3) Crop & segment & measure
        cropped, crop_params = crop_top_and_dish(mask_full, crop_info)
        segmented = segment_plants_from_dish(cropped)

        root_lengths, root_tips, root_paths = {}, {}, {}
        for i, pm in enumerate(segmented, start=1):
            key = f"Plant_{i}"
            length, tip, path = measure_primary_root_and_tip(pm)
            root_lengths[key] = length
            root_tips[key] = tip
            root_paths[key] = path

        merged_crop = merge_segmented_masks(segmented, crop_params)
        full_mask = reconstruct_full_mask(merged_crop, crop_params)

        tips_full, paths_full = adjust_measurements_to_full(
            root_tips, root_paths, crop_params
        )

        # 4) Render overlay TIFF + get measurements
        measurements, overlay_tiff = render_full_mask_with_roots_tiff(
            full_mask, root_lengths, tips_full, paths_full
        )

        # 5) Save the overlay TIFF to disk
        base = Path(file.filename).stem
        out_tif = OUTPUT_DIR / f"{base}_overlay.tif"
        out_tif.write_bytes(overlay_tiff)

        # 6) Build PNG preview
        png = Image.open(io.BytesIO(overlay_tiff)).convert("RGBA")
        png_buf = io.BytesIO()
        png.save(png_buf, format="PNG")
        png_buf.seek(0)

        # 7) Return PNG inline
        return StreamingResponse(
            png_buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={base}_overlay_preview.png"}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
