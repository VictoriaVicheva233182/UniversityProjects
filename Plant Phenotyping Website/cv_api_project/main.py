from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from PIL import Image
import io
import os

# Local imports
from app.endpoint_segment import router as segment_router
from app.endpoint_analyze_mask import router as analyze_mask_router
from app.helper_functions import (
    get_model,  # Proper model loading function
    segment_image,  # Core segmentation logic
)

# --- Load model ---
model = get_model()

# --- Initialize FastAPI app ---
app = FastAPI(title="Plant Phenotyping API")

# --- Register routers ---
app.include_router(segment_router, prefix="/segment", tags=["Segmentation"])
app.include_router(analyze_mask_router, prefix="/analyze-mask", tags=["Mask Analysis"])


# --- Default prediction endpoint ---
@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        # Save input image for debugging
        os.makedirs("debug_inputs", exist_ok=True)
        debug_img = Image.open(io.BytesIO(contents))
        if debug_img.mode in ("RGBA", "P"):
            debug_img = debug_img.convert("RGB")
        debug_img.save("debug_inputs/input_debug.jpg")

        # Run segmentation pipeline
        result = segment_image(model, contents, patch_size=256, step=128)
        mask = result["mask"]

        # Binarize the output mask
        binary_mask = (mask > 0.3).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask, mode="L")

        # Encode and return as PNG
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("🔥 Error during prediction:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
