# app/main.py
from fastapi import FastAPI
from app.routers.segmentation import router as segmentation_router
from app.routers.image_analysis import router as image_router
from app.routers.root_analysis import router as mask_router

app = FastAPI(
    title="Plant Root Analysis API",
    description="Three endpoints for segmentation, mask‐based analysis, and root overlay.",
    version="1.0.0",
)

# mount each of your routers
app.include_router(segmentation_router)
app.include_router(mask_router)
app.include_router(image_router)


@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok"}
