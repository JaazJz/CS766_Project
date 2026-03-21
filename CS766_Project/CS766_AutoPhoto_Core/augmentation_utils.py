import json
from pathlib import Path

import cv2
import numpy as np


ASPECT_RATIOS = {
    "original": None,
    "1:1": 1.0,
    "4:5": 4.0 / 5.0,
    "5:4": 5.0 / 4.0,
    "3:2": 3.0 / 2.0,
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
}


def normalize_map(values):
    values = values.astype(np.float32)
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max - v_min < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - v_min) / (v_max - v_min)


def compute_subject_saliency(image, depth_map):
    """Heuristic subject prior from depth, local contrast, saturation, and center bias."""
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

    depth_norm = normalize_map(depth_map)
    near_prior = 1.0 - depth_norm

    blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
    local_contrast = normalize_map(np.abs(gray - blur))
    edge_strength = normalize_map(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))
    saturation = normalize_map(hsv[:, :, 1])

    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(width - 1, 1)
    yy = yy / max(height - 1, 1)
    center_bias = np.exp(-(((xx - 0.5) ** 2) / 0.10 + ((yy - 0.45) ** 2) / 0.16))
    center_bias = normalize_map(center_bias)

    saliency = (
        0.42 * near_prior
        + 0.20 * local_contrast
        + 0.16 * edge_strength
        + 0.10 * saturation
        + 0.12 * center_bias
    )
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), 5.0)
    return normalize_map(saliency)


def extract_subject_mask(saliency_map):
    """Find the dominant salient region and return a cleaned mask and bounding box."""
    threshold = max(float(np.percentile(saliency_map, 82)), 0.45)
    binary = (saliency_map >= threshold).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_mask = None
    best_bbox = None
    best_score = -1.0
    image_area = saliency_map.shape[0] * saliency_map.shape[1]

    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        if area < max(128, image_area * 0.002):
            continue

        component_mask = (labels == component_id).astype(np.uint8)
        mean_saliency = float(saliency_map[component_mask > 0].mean())
        compactness_bonus = min(area / max(w * h, 1), 1.0)
        score = mean_saliency * (0.7 + 0.3 * compactness_bonus) * np.sqrt(area)
        if score > best_score:
            best_score = score
            best_mask = component_mask
            best_bbox = (int(x), int(y), int(w), int(h))

    if best_mask is None:
        best_mask = np.zeros_like(binary)
        max_y, max_x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        radius = max(24, min(saliency_map.shape[:2]) // 8)
        cv2.circle(best_mask, (int(max_x), int(max_y)), int(radius), 1, thickness=-1)
        best_bbox = _bbox_from_mask(best_mask)

    refined = cv2.GaussianBlur(best_mask.astype(np.float32), (0, 0), 9.0)
    refined = normalize_map(refined)
    return refined, best_bbox


def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return (0, 0, width, height)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def estimate_focus_distance(depth_map, subject_mask, saliency_map):
    weights = (0.7 * normalize_map(subject_mask) + 0.3 * normalize_map(saliency_map)).astype(np.float32)
    active = weights > 0.1
    if not np.any(active):
        return float(np.median(depth_map))

    depths = depth_map[active]
    weights = weights[active]
    order = np.argsort(depths)
    depths = depths[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.4 * cumulative[-1]
    index = int(np.searchsorted(cumulative, cutoff))
    index = min(index, len(depths) - 1)
    return float(depths[index])


def compute_crop_box(image_shape, subject_mask, subject_bbox, aspect_ratio_name):
    height, width = image_shape[:2]
    ratio = ASPECT_RATIOS.get(aspect_ratio_name)
    if ratio is None:
        return (0, 0, width, height)

    x, y, box_w, box_h = subject_bbox
    ys, xs = np.where(subject_mask > 0.15)
    if len(xs) and len(ys):
        subject_cx = float(xs.mean())
        subject_cy = float(ys.mean())
    else:
        subject_cx = x + box_w / 2.0
        subject_cy = y + box_h / 2.0

    pad_x = max(box_w * 0.55, width * 0.08)
    pad_y = max(box_h * 0.60, height * 0.08)
    min_crop_w = min(width, box_w + 2.0 * pad_x)
    min_crop_h = min(height, box_h + 2.0 * pad_y)

    crop_w = max(min_crop_w, min_crop_h * ratio)
    crop_h = crop_w / ratio
    if crop_h > height:
        crop_h = float(height)
        crop_w = crop_h * ratio
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / ratio

    target_x_fraction = 0.36 if subject_cx < width * 0.5 else 0.64
    target_y_fraction = 0.40 if subject_cy < height * 0.5 else 0.56

    crop_x = subject_cx - crop_w * target_x_fraction
    crop_y = subject_cy - crop_h * target_y_fraction
    crop_x = float(np.clip(crop_x, 0, max(width - crop_w, 0)))
    crop_y = float(np.clip(crop_y, 0, max(height - crop_h, 0)))

    crop_x = int(round(crop_x))
    crop_y = int(round(crop_y))
    crop_w = int(round(crop_w))
    crop_h = int(round(crop_h))

    crop_w = min(crop_w, width - crop_x)
    crop_h = min(crop_h, height - crop_y)
    return (crop_x, crop_y, crop_w, crop_h)


def crop_array(values, crop_box):
    x, y, w, h = crop_box
    if values.ndim == 2:
        return values[y:y + h, x:x + w]
    return values[y:y + h, x:x + w, ...]


def apply_style_enhancement(image, subject_mask=None):
    """Lightweight subject-aware tone, color, and detail enhancement."""
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(np.clip(enhanced * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08 + 3.0, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    base = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = np.clip(enhanced + 0.45 * (enhanced - base), 0.0, 1.0)

    if subject_mask is None:
        return sharpened

    mask = normalize_map(subject_mask)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 7.0)
    mask = mask[:, :, np.newaxis]
    return np.clip(enhanced * (1.0 - mask) + sharpened * mask, 0.0, 1.0)


def create_overlay(image, saliency_map, subject_bbox=None, crop_box=None):
    heatmap = cv2.applyColorMap((normalize_map(saliency_map) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * image + 0.45 * heatmap, 0.0, 1.0)
    canvas = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    if subject_bbox is not None:
        x, y, w, h = subject_bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(canvas, "Subject", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if crop_box is not None:
        x, y, w, h = crop_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(canvas, "Crop", (x, min(canvas.shape[0] - 12, y + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return canvas.astype(np.float32) / 255.0


def _fit_height(image, target_height):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def create_summary_panel(original, overlay, final):
    target_height = 480
    panels = [_fit_height(panel, target_height) for panel in (original, overlay, final)]
    labels = ("Original", "Auto Subject", "Augmented")
    labeled_panels = []

    for panel, label in zip(panels, labels):
        panel8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        cv2.rectangle(panel8, (0, 0), (panel8.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(panel8, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        labeled_panels.append(panel8.astype(np.float32) / 255.0)

    return np.hstack(labeled_panels)


def metadata_to_json(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
