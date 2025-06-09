"""
Plant‐Root CLI

Usage:

  # 1) Segment an image:
  python -m cli.main segment <infile> --model-file <model.h5> --out-dir <masks_dir>

  # 2) Process mask TIFF:
  python -m cli.main process-mask <mask.tif> --out-dir <labelled_masks_dir>

  # 3) Full pipeline + overlay:
  python -m cli.main analyze-image <infile> --model-file <model.h5> --out-dir <images_dir>
"""

# ─── Imports ──────────────────────────────────────────────────────────

import io
from pathlib import Path
import typer
import numpy as np
from PIL import Image

from app.utils.io_utils import (
    load_image_from_bytes,
    mask_to_tiff_bytes,
    save_mask_tif,
    load_mask_tif
)
from app.utils.model_utils import load_model
from app.utils.segmentation_utils import (
    segment_image,
    segment_plants_from_dish,
    merge_segmented_masks,
    reconstruct_full_mask
)
from app.utils.postprocessing_utils import (
    threshold_mask,
    morphological_closing,
    crop_top_and_dish
)
from app.utils.analysis_utils import (
    measure_primary_root_and_tip,
    adjust_measurements_to_full,
    render_full_mask_with_roots_tiff,
    overlay_roots_on_image
)

# ─── CLI setup ────────────────────────────────────────────────────────

typer_app = typer.Typer()

# ─── CLI endpoint 1 ───────────────────────────────────────────────────

@typer_app.command("segment")
def cli_segment(
    infile: Path = typer.Argument(..., exists=True, help="Input image file"),
    model_file: Path = typer.Option(..., exists=True, help="Trained U-Net .h5 model"),
    threshold: float = typer.Option(0.1, help="Binarization threshold"),
    out_dir: Path = typer.Option(Path("masks_dir"), help="Output directory for masks")
):
    """
    Segment an image into a binary mask, save TIFF and preview PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(model_file))

    # Load image and segment
    img_bytes = infile.read_bytes()
    seg = segment_image(model, img_bytes)
    raw_mask = seg["mask"]

    # Binarize and refine
    binary = threshold_mask(raw_mask, threshold)
    closed = morphological_closing(binary)

    # Prepare metadata and save TIFF
    mask_uint8 = (closed * 255).astype(np.uint8)
    metadata = {"crop_info": seg.get("crop_info"), "pad_info": seg.get("pad_info")}
    tiff_bytes = mask_to_tiff_bytes(mask_uint8, metadata)
    base = infile.stem
    mask_path = out_dir / f"{base}_mask.tif"
    mask_path.write_bytes(tiff_bytes)

    # Save preview PNG
    png = Image.fromarray(mask_uint8, mode="L")
    preview_path = out_dir / f"{base}_mask_preview.png"
    png.save(preview_path)

    typer.echo(f"Mask TIFF written to {mask_path}")
    typer.echo(f"Mask preview PNG written to {preview_path}")

# ─── CLI endpoint 2 ───────────────────────────────────────────────────

@typer_app.command("process-mask")
def cli_process_mask(
    infile: Path = typer.Argument(..., exists=True, help="Input mask TIFF file"),
    out_dir: Path = typer.Option(Path("masks_labelled_dir"), help="Output directory for labelled masks")
):
    """
    Process a mask TIFF: crop, measure roots, overlay, save TIFF & PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask, meta = load_mask_tif(str(infile))
    crop_info = meta.get("crop_info")
    if crop_info is None:
        typer.secho("ERROR: mask TIFF missing crop_info", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Crop, segment, measure
    cropped, params = crop_top_and_dish(mask, crop_info)
    segments = segment_plants_from_dish(cropped)
    lengths, tips, paths = {}, {}, {}
    for i, seg_mask in enumerate(segments, start=1):
        key = f"Plant_{i}"
        l, t, p = measure_primary_root_and_tip(seg_mask)
        lengths[key], tips[key], paths[key] = l, t, p

    merged = merge_segmented_masks(segments, params)
    full_mask = reconstruct_full_mask(merged, params)
    tips_full, paths_full = adjust_measurements_to_full(tips, paths, params)

    measurements, tiff_bytes = render_full_mask_with_roots_tiff(
        full_mask, lengths, tips_full, paths_full
    )
    base = infile.stem
    out_tif = out_dir / f"{base}_overlay.tif"
    out_tif.write_bytes(tiff_bytes)

    # Save preview PNG
    png = Image.open(io.BytesIO(tiff_bytes)).convert("RGBA")
    preview_path = out_dir / f"{base}_overlay_preview.png"
    png.save(preview_path)

    typer.echo(f"Overlay TIFF written to {out_tif}")
    typer.echo(f"Overlay preview PNG written to {preview_path}")

# ─── CLI endpoint 3 ───────────────────────────────────────────────────

@typer_app.command("analyze-image")
def cli_analyze_image(
    infile: Path = typer.Argument(..., exists=True, help="Input image file"),
    model_file: Path = typer.Option(..., exists=True, help="Trained U-Net .h5 model"),
    threshold: float = typer.Option(0.1, help="Mask binarization threshold"),
    out_dir: Path = typer.Option(Path("images_labelled_dir"), help="Output directory for labelled images")
):
    """
    Full pipeline: segment image, measure roots, overlay on original, save PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(model_file))

    # Segment
    img_bytes = infile.read_bytes()
    seg = segment_image(model, img_bytes)
    raw_mask = seg["mask"]
    binary = threshold_mask(raw_mask, threshold)
    closed = morphological_closing(binary)
    closed_u8 = (closed * 255).astype(np.uint8)

    crop_info = seg.get("crop_info")
    if crop_info is None:
        typer.secho("ERROR: segmentation missing crop_info", fg=typer.colors.RED)
        raise typer.Exit(1)
    cropped, params = crop_top_and_dish(closed_u8, crop_info)
    segments = segment_plants_from_dish(cropped)
    lengths, tips, paths = {}, {}, {}
    for i, seg_mask in enumerate(segments, start=1):
        key = f"Plant_{i}"
        l, t, p = measure_primary_root_and_tip(seg_mask)
        lengths[key], tips[key], paths[key] = l, t, p

    tips_full, paths_full = adjust_measurements_to_full(tips, paths, params)
    overlay_img, _ = overlay_roots_on_image(
        img_bytes, tips_full, paths_full, lengths
    )

    base = infile.stem
    out_png = out_dir / f"{base}_overlay.png"
    overlay_img.save(out_png)
    typer.echo(f"Overlay image written to {out_png}")

# ─── Main entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    typer_app()
 