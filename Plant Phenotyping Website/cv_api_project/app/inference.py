import io
import cv2
import numpy as np
import networkx as nx
from PIL import Image
from skimage.morphology import skeletonize
from typing import Optional, Tuple, Dict, List


def compute_root_length(plant_mask: np.ndarray) -> float:
    binary = (plant_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return 0.0
    largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    comp = (labels == largest).astype(np.uint8)
    x, y, w, h, _ = stats[largest]
    crop = comp[y : y + h, x : x + w]
    skel = skeletonize(crop)
    coords = np.argwhere(skel)
    if coords.size == 0:
        return 0.0
    G = nx.Graph()
    pts = set(map(tuple, coords))
    for r, c in pts:
        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nbr = (r + dr, c + dc)
            if nbr in pts:
                G.add_edge((r, c), nbr, weight=np.hypot(dr, dc))
    if G.number_of_nodes() == 0:
        return 0.0
    top = min(G.nodes, key=lambda n: n[0])
    bot = max(G.nodes, key=lambda n: n[0])
    try:
        return nx.dijkstra_path_length(G, source=top, target=bot, weight="weight")
    except nx.NetworkXNoPath:
        return 0.0


def compute_tip_coordinate(
    plant_mask: np.ndarray, y_offset: int
) -> Optional[Tuple[int, int]]:
    skel = skeletonize(plant_mask.astype(bool)).astype(np.uint8)
    coords = np.argwhere(skel)
    if coords.size == 0:
        return None

    G = nx.Graph()
    pts = set(map(tuple, coords))
    for r, c in pts:
        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nbr = (r + dr, c + dc)
            if nbr in pts:
                G.add_edge((r, c), nbr, weight=np.hypot(dr, dc))

    if not G:
        return None

    top_point = min(G.nodes, key=lambda x: x[0])
    lengths, _ = nx.single_source_dijkstra(G, source=top_point)
    furthest_point = max(lengths, key=lengths.get)

    return (int(furthest_point[1]), int(furthest_point[0]))


def annotate_tips_and_lengths(
    dish_mask: np.ndarray, tips: List[Optional[Tuple[int, int]]], lengths: List[float]
) -> np.ndarray:
    canvas = cv2.cvtColor((dish_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i, tip in enumerate(tips):
        if tip is None:
            continue
        x, y = tip
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"{lengths[i]:.1f}px",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def mask_to_png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 3 and img.shape[2] == 3:
        mode = "RGB"
    else:
        mode = "L"
    pil = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def combine_plant_masks_labeled(plant_masks: List[np.ndarray]) -> np.ndarray:
    h, w = plant_masks[0].shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    for idx, pm in enumerate(plant_masks, start=1):
        label_mask[pm.astype(bool)] = idx
    return label_mask


def gather_plant_data(
    dish_top: np.ndarray, y_offset: int, num_plants: int = 5, **segment_kwargs
) -> Tuple[Dict[int, Dict], List[np.ndarray]]:
    from app.postprocessing import (
        segment_roots,
    )  # local import to avoid circular import

    plants = segment_roots(dish_top, num_plants=num_plants, **segment_kwargs)
    data = {}
    for i, pm in enumerate(plants, start=1):
        tip = compute_tip_coordinate(pm, y_offset)
        length = compute_root_length(pm)
        data[i] = {"mask": pm, "tip": tip, "length": length}
    return data, plants


def annotate_labeled_mask_with_tips(
    label_mask: np.ndarray, plant_data: Dict[int, Dict]
) -> np.ndarray:
    canvas = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    max_label = label_mask.max()
    for lab in range(1, max_label + 1):
        shade = int(255 * lab / (max_label + 1))
        canvas[label_mask == lab] = (shade, shade, shade)

    for i, info in plant_data.items():
        tip = info["tip"]
        if tip is None:
            continue
        x, y = tip
        cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
        txt = f"{i}:{info['length']:.1f}px"
        cv2.putText(
            canvas,
            txt,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def map_tip_to_original(
    tip: Optional[Tuple[int, int]], crop_info: Dict[str, int], y_offset: int
) -> Optional[Tuple[int, int]]:
    if tip is None:
        return None
    tx, ty = tip
    x0, y0 = crop_info["x_start"], crop_info["y_start"]
    return (tx + x0, ty + y0 + y_offset)


def annotate_gray_image_with_tips(
    gray_img: np.ndarray,
    mapped_tips: List[Optional[Tuple[int, int]]],
    lengths: List[float],
) -> np.ndarray:
    canvas = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for i, tip in enumerate(mapped_tips, start=1):
        if tip is None:
            continue
        x, y = tip
        cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
        label = f"{i}:{lengths[i - 1]:.1f}px"
        cv2.putText(
            canvas,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
