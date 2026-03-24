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


def _detect_people_hog(rgb8):
    """HOG person detector. Returns list of (x, y, w, h, conf) in original coords."""
    height, width = rgb8.shape[:2]
    scale = min(1.0, 800 / max(height, width))
    small_w = max(64, int(round(width * scale)))
    small_h = max(64, int(round(height * scale)))
    small = cv2.resize(rgb8, (small_w, small_h), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        cv2.cvtColor(small, cv2.COLOR_RGB2GRAY),
        winStride=(8, 8), padding=(16, 16), scale=1.05,
    )
    detections = []
    if len(rects):
        for (rx, ry, rw, rh), w in zip(rects, weights):
            detections.append((int(rx/scale), int(ry/scale),
                               int(rw/scale), int(rh/scale),
                               float(np.clip(w, 0.0, 2.0))))
    return detections


def _people_prominence(detections, img_width, img_height):
    """Score [0,1]: how much are people the intended subject vs. the scene."""
    if not detections:
        return 0.0
    img_area = img_width * img_height
    edge_margin = 0.12
    total_af, cx_list, cy_list, conf_list, clip_pen = 0.0, [], [], [], 0.0
    for (x, y, w, h, conf) in detections:
        af = (w * h) / img_area
        total_af += af
        cx_list.append((x + w / 2) / img_width)
        cy_list.append((y + h / 2) / img_height)
        conf_list.append(conf)
        if (x < img_width * edge_margin
                or (x + w) > img_width * (1 - edge_margin)
                or y < img_height * edge_margin
                or (y + h) > img_height * (1 - edge_margin)):
            clip_pen += af
    dist = np.sqrt((np.mean(cx_list) - 0.5)**2 + (np.mean(cy_list) - 0.5)**2)
    centrality  = float(np.clip(1.0 - dist * 2.0, 0.0, 1.0))
    area_score  = float(np.clip((total_af - 0.03) / 0.12, 0.0, 1.0))
    clip_factor = float(np.clip(1.0 - clip_pen * 4.0, 0.0, 1.0))
    mean_conf   = float(np.mean(conf_list))
    p = (area_score * 0.45 + centrality * 0.35 + (mean_conf / 2.0) * 0.20)
    return float(np.clip(p * clip_factor, 0.0, 1.0))


def compute_subject_saliency(image, depth_map, subject_mode="auto"):
    """
    Adaptive saliency map with three modes:

    subject_mode='scene'  – depth + contrast heuristic only (good for
                            landscapes, architecture, street environments).
    subject_mode='people' – HOG person heatmap dominates regardless of size.
    subject_mode='auto'   – decides automatically via _people_prominence:
                            < 0.25 → scene, 0.25-0.55 → blend, > 0.55 → people.
    """
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv  = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    height, width = gray.shape

    # ── Scene signals (original weights, unchanged) ───────────────────────────
    near_prior     = normalize_map(1.0 - normalize_map(depth_map))
    blur           = cv2.GaussianBlur(gray, (0, 0), 5.0)
    local_contrast = normalize_map(np.abs(gray - blur))
    edge_strength  = normalize_map(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))
    saturation     = normalize_map(hsv[:, :, 1])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(width - 1, 1);  yy /= max(height - 1, 1)
    center_bias = normalize_map(
        np.exp(-(((xx - 0.5)**2) / 0.10 + ((yy - 0.45)**2) / 0.16))
    )
    scene_sal = (0.42 * near_prior + 0.20 * local_contrast
                 + 0.16 * edge_strength + 0.10 * saturation + 0.12 * center_bias)

    # ── Determine effective blend weight ─────────────────────────────────────
    detections = _detect_people_hog(rgb8)
    prominence = _people_prominence(detections, width, height)

    if subject_mode == "scene":
        effective_alpha = 0.0
    elif subject_mode == "people":
        effective_alpha = 1.0 if detections else 0.0
    else:  # auto
        if prominence > 0.55:
            effective_alpha = 1.0
        elif prominence > 0.25:
            effective_alpha = (prominence - 0.25) / 0.30
        else:
            effective_alpha = 0.0

    mode_label = (subject_mode if subject_mode != "auto"
                  else ("people" if prominence > 0.55
                        else "blend" if prominence > 0.25 else "scene"))
    print(f"  Subject mode: {mode_label}  "
          f"(prominence={prominence:.2f}, {len(detections)} detection(s))")

    if effective_alpha < 0.01 or not detections:
        saliency = scene_sal
    else:
        person_map = np.zeros((height, width), dtype=np.float32)
        for (px, py, pw, ph, conf) in detections:
            cx, cy = px + pw / 2.0, py + ph / 2.0
            sx, sy = max(pw * 0.35, 8.0), max(ph * 0.35, 8.0)
            yy2, xx2 = np.mgrid[0:height, 0:width].astype(np.float32)
            blob = conf * np.exp(-(((xx2 - cx)**2) / (2*sx**2)
                                   + ((yy2 - cy)**2) / (2*sy**2)))
            person_map = np.maximum(person_map, blob)
        person_map = normalize_map(person_map)
        people_sal = (0.55 * person_map + 0.25 * near_prior
                      + 0.10 * local_contrast + 0.06 * edge_strength
                      + 0.04 * saturation)
        saliency = (1.0 - effective_alpha) * scene_sal + effective_alpha * people_sal

    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), 5.0)
    return normalize_map(saliency)


def extract_subject_mask(saliency_map):
    height, width = saliency_map.shape[:2]
    img_area = height * width
    img_min_dim = min(height, width)

    threshold = max(float(np.percentile(saliency_map, 82)), 0.45)
    binary = (saliency_map >= threshold).astype(np.uint8)

    k = max(3, img_min_dim // 140)
    k = k if k % 2 == 1 else k + 1          # must be odd for symmetry
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_mask = None
    best_bbox = None
    best_score = -1.0

    min_area = max(128, img_area * 0.005)

    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        if area < min_area:
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
        # fallback circle also scales with image size
        radius = max(24, img_min_dim // 10)
        cv2.circle(best_mask, (int(max_x), int(max_y)), int(radius), 1, thickness=-1)
        best_bbox = _bbox_from_mask(best_mask)

    blur_sigma = max(5.0, img_min_dim * 0.012)
    refined = cv2.GaussianBlur(best_mask.astype(np.float32), (0, 0), blur_sigma)
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

    subject_ref = max(box_w, box_h)
    crop_short = max(subject_ref / 0.38, min(width, height) * 0.65)
    crop_short = min(crop_short, min(width, height))

    if ratio <= 1.0:
        crop_w = crop_short
        crop_h = crop_w / ratio
    else:
        crop_h = crop_short
        crop_w = crop_h * ratio

    if crop_w > width:
        crop_w = float(width);  crop_h = crop_w / ratio
    if crop_h > height:
        crop_h = float(height); crop_w = crop_h * ratio

    sx_norm = subject_cx / max(width  - 1, 1)
    sy_norm = subject_cy / max(height - 1, 1)
    anchor_x = 0.50 + (sx_norm - 0.50) * 0.20
    anchor_y = 0.50 + (sy_norm - 0.50) * 0.16

    crop_x = subject_cx - crop_w * anchor_x
    crop_y = subject_cy - crop_h * anchor_y
    crop_x = float(np.clip(crop_x, 0, max(width  - crop_w, 0)))
    crop_y = float(np.clip(crop_y, 0, max(height - crop_h, 0)))

    crop_x = int(round(crop_x))
    crop_y = int(round(crop_y))
    crop_w = int(round(min(crop_w, width  - crop_x)))
    crop_h = int(round(min(crop_h, height - crop_y)))
    return (crop_x, crop_y, crop_w, crop_h)


def crop_array(values, crop_box):
    x, y, w, h = crop_box
    if values.ndim == 2:
        return values[y:y + h, x:x + w]
    return values[y:y + h, x:x + w, ...]


def _unsharp_mask(image, sigma=1.2, amount=0.45):
    base = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + amount * (image - base), 0.0, 1.0)


def _soft_subject_mask(subject_mask):
    mask = normalize_map(subject_mask)
    mask = np.clip((mask - 0.12) / 0.50, 0.0, 1.0)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 9.0)


def _subject_center(subject_mask):
    mask = _soft_subject_mask(subject_mask)
    height, width = mask.shape
    total = float(mask.sum())
    if total < 1e-6:
        return width / 2.0, height / 2.0

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = float((xx * mask).sum() / total)
    center_y = float((yy * mask).sum() / total)
    return center_x, center_y


def preserve_subject_focus(original, refocused, subject_mask):
    mask = _soft_subject_mask(subject_mask)
    mask = mask[:, :, np.newaxis]

    crisp_subject = _unsharp_mask(original, sigma=1.0, amount=0.35)
    protected = refocused * (1.0 - mask) + crisp_subject * mask
    return np.clip(protected, 0.0, 1.0)


def enhance_subject_separation(image, subject_mask):
    mask = _soft_subject_mask(subject_mask)
    mask3 = mask[:, :, np.newaxis]

    background_soft = cv2.GaussianBlur(image, (0, 0), 2.4)
    background_weight = np.clip((1.0 - mask) ** 1.6 * 0.55, 0.0, 0.55)
    separated = image * (1.0 - background_weight[:, :, np.newaxis]) + background_soft * background_weight[:, :, np.newaxis]

    center_x, center_y = _subject_center(subject_mask)
    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = (xx - center_x) / max(width * 0.42, 1.0)
    dy = (yy - center_y) / max(height * 0.42, 1.0)
    radial = np.sqrt(dx * dx + dy * dy)
    vignette = np.clip(1.04 - 0.16 * radial, 0.84, 1.04)
    vignette = vignette[:, :, np.newaxis]

    subject_lift = 1.0 + 0.06 * mask3
    return np.clip(separated * vignette * subject_lift, 0.0, 1.0)


def create_augmented_image(original, refocused, subject_mask):
    protected = preserve_subject_focus(original, refocused, subject_mask)
    separated = enhance_subject_separation(protected, subject_mask)
    return apply_style_enhancement(separated, subject_mask)


def apply_style_enhancement(image, subject_mask=None):
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

    sharpened = _unsharp_mask(enhanced, sigma=1.2, amount=0.45)

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
