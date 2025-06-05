from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Union, Tuple, Dict, Optional, List, Any
import io
import csv
import json
import base64

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import networkx as nx
from skimage.morphology import skeletonize

router = APIRouter()

# --- Utility Functions ---


def load_mask_tif(mask_src: Union[str, io.BytesIO]) -> Tuple[np.ndarray, Dict]:
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        desc = tif.pages[0].tags["ImageDescription"].value
    metadata = json.loads(desc) if desc else {}
    return mask, metadata


def crop_top_and_dish(
    binary: np.ndarray, crop_info: Dict[str, Any], top_crop_ratio: float = 0.15
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x_start = crop_info.get("x_start", 0)
    crop_size = crop_info.get("crop_size", binary.shape[1])
    x_end = min(x_start + crop_size, binary.shape[1])
    dish = binary[:, x_start:x_end]
    top_crop = int(dish.shape[0] * top_crop_ratio)
    cropped = dish[top_crop:, :]
    params = {
        "x_start": x_start,
        "x_end": x_end,
        "top_crop": top_crop,
        "orig_shape": binary.shape,
    }
    return cropped, params


def segment_plants_from_dish(
    mask: np.ndarray,
    num_plants: int = 5,
    min_area: int = 100,
    aspect_ratio_threshold: float = 1.5,
    vertical_start_thresh_ratio: float = 0.3,
    angle_filter_area_thresh: int = 1200,
    min_angle_degrees: float = 65,
    edge_margin_px: int = 250,
) -> List[np.ndarray]:
    h, w = mask.shape
    band_width = w / num_plants
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    plant_masks: List[np.ndarray] = [
        np.zeros_like(mask, dtype=np.uint8) for _ in range(num_plants)
    ]

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]
        if (
            area < min_area
            or (cw > 0 and ch / cw < aspect_ratio_threshold)
            or y > int(h * vertical_start_thresh_ratio)
        ):
            continue
        if area < angle_filter_area_thresh:
            angle = np.degrees(np.arctan2(ch, cw))
            if (
                angle < min_angle_degrees
                or x < edge_margin_px
                or (x + cw) > (w - edge_margin_px)
            ):
                continue
        band_idx = min(int(cx // band_width), num_plants - 1)
        component_mask = (labels == i).astype(np.uint8)
        if np.count_nonzero(component_mask) > np.count_nonzero(plant_masks[band_idx]):
            plant_masks[band_idx] = component_mask

    return [(m * 255).astype(np.uint8) for m in plant_masks]


def measure_primary_root_and_tip(
    mask: np.ndarray,
) -> Tuple[float, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    binary = (mask > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels.max() == 0:
        return 0.0, None, []
    i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, _ = stats[i]
    comp = (labels == i).astype(np.uint8)
    ske = skeletonize(comp[y : y + h, x : x + w].astype(bool)).astype(np.uint8)
    pts = set(map(tuple, np.argwhere(ske)))
    G = nx.Graph()
    for ry, rx in pts:
        for dy, dx in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nb = (ry + dy, rx + dx)
            if nb in pts:
                G.add_edge((ry, rx), nb, weight=np.hypot(dy, dx))
    if G.number_of_nodes() == 0:
        return 0.0, None, []
    top = min(G.nodes, key=lambda n: n[0])
    bot = max(G.nodes, key=lambda n: n[0])
    try:
        path = nx.dijkstra_path(G, top, bot, "weight")
        length = nx.dijkstra_path_length(G, top, bot, "weight")
    except nx.NetworkXNoPath:
        return 0.0, None, []
    full_path = [(ry + y, rx + x) for (ry, rx) in path]
    return round(length, 2), full_path[-1], full_path


def merge_segmented_masks(
    segmented: List[np.ndarray], crop_params: Dict[str, Any]
) -> np.ndarray:
    H_crop, W_crop = segmented[0].shape
    combined = np.zeros((H_crop, W_crop), dtype=np.uint8)
    for m in segmented:
        combined = np.maximum(combined, m)
    return combined


def reconstruct_full_mask(
    cropped_mask: np.ndarray, crop_params: Dict[str, Any]
) -> np.ndarray:
    H_full, W_full = crop_params["orig_shape"]
    x_start = crop_params["x_start"]
    top_crop = crop_params["top_crop"]
    full_mask = np.zeros((H_full, W_full), dtype=np.uint8)
    full_mask[
        top_crop : top_crop + cropped_mask.shape[0],
        x_start : x_start + cropped_mask.shape[1],
    ] = cropped_mask
    return full_mask


def adjust_measurements_to_full(root_tips, root_paths, crop_params):
    x0 = crop_params["x_start"]
    y0 = crop_params["top_crop"]
    tips_full = {}
    paths_full = {}
    for key, tip in root_tips.items():
        if tip is None:
            tips_full[key] = None
            paths_full[key] = []
        else:
            r, c = tip
            tips_full[key] = (r + y0, c + x0)
            path = root_paths.get(key, [])
            paths_full[key] = [(r0 + y0, c0 + x0) for r0, c0 in path]
    return tips_full, paths_full


def render_full_mask_with_roots_tiff(full_mask, root_lengths, tips_full, paths_full):
    measurements = {
        key: {"length_px": length, "tip_coord": tips_full.get(key)}
        for key, length in root_lengths.items()
    }
    desc = json.dumps(measurements)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(full_mask, cmap="gray", vmin=0, vmax=255)
    cmap = plt.get_cmap("tab10")
    for idx, key in enumerate(root_lengths):
        path = paths_full.get(key, [])
        if path:
            y_path, x_path = zip(*path)
            ax.plot(x_path, y_path, color=cmap(idx), linewidth=2)
        tip = tips_full.get(key)
        if tip:
            ty, tx = tip
            ax.scatter(tx, ty, s=100, c="yellow", edgecolors="black", linewidths=1.5)
            ax.text(
                tx + 5,
                ty + 5,
                f"{root_lengths[key]:.1f}px",
                color=cmap(idx),
                fontsize=8,
                weight="bold",
            )
    ax.axis("off")
    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
    img = buf[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    tiff_buf = io.BytesIO()
    tifffile.imwrite(tiff_buf, img, photometric="rgb", description=desc)
    tiff_buf.seek(0)
    return measurements, tiff_buf.read()


def package_analysis_outputs(measurements, paths_full, overlay_tiff):
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["plant", "length_px", "tip_row", "tip_col"])
    for plant, m in measurements.items():
        tip = m.get("tip_coord") or (None, None)
        writer.writerow([plant, m["length_px"], tip[0], tip[1]])
    return {
        "measurements": measurements,
        "paths": paths_full,
        "overlay_tiff": overlay_tiff,
        "measurements_csv": csv_buf.getvalue(),
    }


# --- Endpoint ---
@router.post("/masks/process")
async def process_mask(file: UploadFile = File(...)):
    data = await file.read()
    buf = io.BytesIO(data)
    try:
        mask_full, crop_info = load_mask_tif(buf)
    except Exception as e:
        raise HTTPException(400, f"Invalid TIFF upload: {e}")
    cropped, crop_params = crop_top_and_dish(mask_full, crop_info)
    segmented = segment_plants_from_dish(cropped)

    root_lengths = {}
    root_tips = {}
    root_paths = {}
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

    measurements, tiff_bytes = render_full_mask_with_roots_tiff(
        full_mask, root_lengths, tips_full, paths_full
    )
    package = package_analysis_outputs(measurements, paths_full, tiff_bytes)
    overlay_b64 = base64.b64encode(package["overlay_tiff"]).decode("ascii")

    return JSONResponse(
        {
            "measurements": package["measurements"],
            "paths": package["paths"],
            "measurements_csv": package["measurements_csv"],
            "overlay_tiff_b64": overlay_b64,
        }
    )
