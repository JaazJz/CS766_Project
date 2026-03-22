"""
filter_utils.py  –  Camera / film style filters for the Auto Photo pipeline.

Each style is a bundle of:
  • film grain  (luminance + chroma, spatially correlated)
  • tone curve  (per-channel or global)
  • color grade (shadows / midtones / highlights lift)
  • saturation tweak
  • vignette strength

Public API
----------
apply_film_filter(image, style_name, strength=1.0, seed=None) -> np.ndarray
list_styles() -> list[str]
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

STYLES = {
    # ── COLOUR FILM ─────────────────────────────────────────────────────────
    "kodak_portra": dict(
        grain_luma=0.028, grain_chroma=0.008, grain_sigma=0.55,
        saturation=0.92,
        curve_r=[0, 12, 128, 140, 255, 248],   # (in, out) pairs flat-list
        curve_g=[0, 8,  128, 135, 255, 250],
        curve_b=[0, 5,  128, 128, 255, 245],
        shadow_tint=(+4, +2, -3),               # RGB lift in [0..255] space
        highlight_tint=(+2, 0, -2),
        vignette=0.22,
        description="Kodak Portra 400 – warm skin tones, fine grain, lifted shadows",
    ),
    "fuji_velvia": dict(
        grain_luma=0.018, grain_chroma=0.012, grain_sigma=0.40,
        saturation=1.22,
        curve_r=[0, 0,  128, 132, 255, 255],
        curve_g=[0, 0,  128, 130, 255, 254],
        curve_b=[0, 4,  128, 124, 255, 250],
        shadow_tint=(-2, +1, +6),
        highlight_tint=(0, +1, -3),
        vignette=0.18,
        description="Fuji Velvia 50 – punchy saturation, cool shadows, landscapes",
    ),
    "fuji_pro400h": dict(
        grain_luma=0.022, grain_chroma=0.010, grain_sigma=0.50,
        saturation=0.88,
        curve_r=[0, 8,  128, 130, 255, 245],
        curve_g=[0, 6,  128, 132, 255, 248],
        curve_b=[0, 10, 128, 136, 255, 252],
        shadow_tint=(-2, +2, +8),
        highlight_tint=(0, +2, +4),
        vignette=0.15,
        description="Fuji Pro 400H – pastel/airy, cool cast, popular for portraits",
    ),
    "cinestill_800t": dict(
        grain_luma=0.055, grain_chroma=0.022, grain_sigma=0.70,
        saturation=1.05,
        curve_r=[0, 5,  128, 140, 255, 255],
        curve_g=[0, 2,  128, 128, 255, 248],
        curve_b=[0, 10, 128, 118, 255, 235],
        shadow_tint=(+8, +2, -6),
        highlight_tint=(+4, 0, -8),
        vignette=0.35,
        description="CineStill 800T – tungsten-balanced, halation glow, night / neon",
    ),

    # ── BLACK & WHITE ────────────────────────────────────────────────────────
    "ilford_hp5": dict(
        grain_luma=0.045, grain_chroma=0.0, grain_sigma=0.65,
        saturation=0.0,
        curve_r=[0, 0, 80, 72, 180, 185, 255, 255],
        curve_g=[0, 0, 80, 72, 180, 185, 255, 255],
        curve_b=[0, 0, 80, 72, 180, 185, 255, 255],
        shadow_tint=(0, 0, 0),
        highlight_tint=(0, 0, 0),
        vignette=0.28,
        description="Ilford HP5 – classic B&W, medium grain, wide latitude",
    ),
    "kodak_trix": dict(
        grain_luma=0.065, grain_chroma=0.0, grain_sigma=0.80,
        saturation=0.0,
        curve_r=[0, 0, 60, 48, 190, 200, 255, 255],
        curve_g=[0, 0, 60, 48, 190, 200, 255, 255],
        curve_b=[0, 0, 60, 48, 190, 200, 255, 255],
        shadow_tint=(0, 0, 0),
        highlight_tint=(0, 0, 0),
        vignette=0.38,
        description="Kodak Tri-X – high contrast, coarse grain, gritty documentary",
    ),

    # ── SPECIAL / CINEMATIC ──────────────────────────────────────────────────
    "teal_orange": dict(
        grain_luma=0.020, grain_chroma=0.005, grain_sigma=0.45,
        saturation=1.10,
        curve_r=[0, 0,  128, 138, 255, 255],
        curve_g=[0, 0,  128, 125, 255, 248],
        curve_b=[0, 0,  128, 118, 255, 240],
        shadow_tint=(-6, +4, +10),
        highlight_tint=(+8, +2, -6),
        vignette=0.20,
        description="Teal & Orange – Hollywood blockbuster look",
    ),
    "vintage_faded": dict(
        grain_luma=0.038, grain_chroma=0.015, grain_sigma=0.60,
        saturation=0.78,
        curve_r=[20, 35, 128, 138, 230, 225],
        curve_g=[20, 28, 128, 132, 230, 220],
        curve_b=[20, 32, 128, 125, 230, 210],
        shadow_tint=(+10, +5, +2),
        highlight_tint=(+4, +2, -4),
        vignette=0.42,
        description="Vintage faded – lifted blacks, warm cast, worn feel",
    ),
}


def list_styles():
    """Return sorted list of available style names."""
    return sorted(STYLES.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_lut(curve_pairs_flat):
    """
    Build a uint8 → uint8 LUT from a flat list of (in, out) control points.
    E.g. [0, 5, 128, 135, 255, 248]  →  three knots.
    Uses monotone cubic interpolation so the curve stays smooth.
    """
    pts = np.array(curve_pairs_flat, dtype=np.float32).reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]

    # Ensure endpoints exist
    if xs[0] > 0:
        xs = np.r_[0, xs]
        ys = np.r_[ys[0], ys]
    if xs[-1] < 255:
        xs = np.r_[xs, 255]
        ys = np.r_[ys, ys[-1]]

    all_x = np.arange(256, dtype=np.float32)
    lut = np.interp(all_x, xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _apply_tone_curve(image_f32, curve_r, curve_g, curve_b):
    """Apply per-channel tone curve to float [0,1] image."""
    img8 = np.clip(image_f32 * 255.0, 0, 255).astype(np.uint8)
    lut_r = _build_lut(curve_r)
    lut_g = _build_lut(curve_g)
    lut_b = _build_lut(curve_b)
    out = np.stack([
        lut_r[img8[:, :, 0]],
        lut_g[img8[:, :, 1]],
        lut_b[img8[:, :, 2]],
    ], axis=2)
    return out.astype(np.float32) / 255.0


def _apply_color_grade(image_f32, shadow_tint, highlight_tint):
    """
    Additive shadow / highlight colour tint.
    shadow_tint   – (R, G, B) offset in [0..255] scale, applied in dark regions
    highlight_tint – same, applied in bright regions
    """
    luma = image_f32.mean(axis=2, keepdims=True)           # [0,1]
    shadow_w    = np.clip(1.0 - luma * 3.0, 0.0, 1.0)    # strong in shadows
    highlight_w = np.clip((luma - 0.6) * 3.0, 0.0, 1.0)  # strong in highlights

    s = np.array(shadow_tint,    dtype=np.float32) / 255.0
    h = np.array(highlight_tint, dtype=np.float32) / 255.0

    out = image_f32 + shadow_w * s + highlight_w * h
    return np.clip(out, 0.0, 1.0)


def _adjust_saturation(image_f32, factor):
    """Scale saturation.  factor=0 → greyscale, 1 → unchanged, >1 → boosted."""
    if abs(factor - 1.0) < 1e-4 and factor > 0.05:
        return image_f32
    gray = image_f32.mean(axis=2, keepdims=True)
    if factor < 0.01:   # full B&W – use perceptual weights instead
        gray = (0.299 * image_f32[:, :, 0:1]
              + 0.587 * image_f32[:, :, 1:2]
              + 0.114 * image_f32[:, :, 2:3])
        return np.repeat(gray, 3, axis=2)
    out = gray + factor * (image_f32 - gray)
    return np.clip(out, 0.0, 1.0)


def _add_film_grain(image_f32, luma_amount, chroma_amount, sigma, seed=None):
    """
    Spatially-correlated luminance + chroma grain (much closer to real film than
    plain Gaussian noise).

    Method:
      1. Generate white noise then blur to the desired grain size (sigma controls
         how 'clumpy' the grain is – large sigma = coarse grain).
      2. Scale to target RMS amplitude.
      3. Add luminance grain uniformly; add chroma grain in Lab space.
    """
    if luma_amount < 1e-4 and chroma_amount < 1e-4:
        return image_f32

    rng = np.random.default_rng(seed)
    H, W = image_f32.shape[:2]

    def _grain_layer(amount, color_channels=1):
        raw = rng.standard_normal((H, W, color_channels)).astype(np.float32)
        # Blur for spatial correlation
        blurred = np.stack([
            cv2.GaussianBlur(raw[:, :, c], (0, 0), sigma)
            for c in range(color_channels)
        ], axis=2)
        # Normalise to unit variance then scale
        std = blurred.std() + 1e-8
        return blurred / std * amount

    result = image_f32.copy()

    # Luminance grain – add in linear RGB (affects all channels equally)
    if luma_amount > 1e-4:
        lg = _grain_layer(luma_amount, 1)          # (H, W, 1)
        result = np.clip(result + lg, 0.0, 1.0)

    # Chroma grain – perturb in Lab a/b channels only
    if chroma_amount > 1e-4:
        lab = cv2.cvtColor(
            np.clip(result * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        cg = _grain_layer(chroma_amount * 128.0, 2)   # a, b channels ±
        lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] + cg[:, :, :], 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    return result


def _add_vignette(image_f32, strength, subject_cx=None, subject_cy=None):
    """
    Smooth radial vignette.  If subject center is provided, the vignette is
    shifted slightly toward the subject for a natural look.
    """
    if strength < 1e-4:
        return image_f32

    H, W = image_f32.shape[:2]
    cx = subject_cx if subject_cx is not None else W / 2.0
    cy = subject_cy if subject_cy is not None else H / 2.0

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - cx) / (W * 0.50)
    dy = (yy - cy) / (H * 0.50)
    radial = np.sqrt(dx * dx + dy * dy)
    vignette = 1.0 - np.clip(strength * (radial ** 1.8), 0.0, strength)
    return np.clip(image_f32 * vignette[:, :, np.newaxis], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_film_filter(
    image: np.ndarray,
    style_name: str,
    strength: float = 1.0,
    subject_cx: float = None,
    subject_cy: float = None,
    seed: int = None,
) -> np.ndarray:
    """
    Apply a named film / camera style filter to a float RGB [0,1] image.

    Parameters
    ----------
    image       : H×W×3 float32 in [0, 1]
    style_name  : one of list_styles()
    strength    : 0.0 = no effect, 1.0 = full style (grain & grade linearly scaled)
    subject_cx/cy : optional subject centre for vignette anchoring (pixels)
    seed        : RNG seed for reproducible grain

    Returns
    -------
    Filtered H×W×3 float32 in [0, 1]
    """
    if style_name not in STYLES:
        raise ValueError(f"Unknown style '{style_name}'. Available: {list_styles()}")

    s = STYLES[style_name]
    result = image.astype(np.float32).copy()

    # 1. Tone curve
    result = _apply_tone_curve(result, s["curve_r"], s["curve_g"], s["curve_b"])

    # 2. Saturation
    sat_target = 1.0 + (s["saturation"] - 1.0) * strength
    result = _adjust_saturation(result, sat_target)

    # 3. Color grade
    sh = tuple(v * strength for v in s["shadow_tint"])
    hi = tuple(v * strength for v in s["highlight_tint"])
    result = _apply_color_grade(result, sh, hi)

    # 4. Film grain
    result = _add_film_grain(
        result,
        luma_amount  = s["grain_luma"]   * strength,
        chroma_amount= s["grain_chroma"] * strength,
        sigma        = s["grain_sigma"],
        seed         = seed,
    )

    # 5. Vignette
    result = _add_vignette(
        result,
        strength    = s["vignette"] * strength,
        subject_cx  = subject_cx,
        subject_cy  = subject_cy,
    )

    return np.clip(result, 0.0, 1.0)


def apply_filter_batch(
    image: np.ndarray,
    styles: list,
    strength: float = 1.0,
    subject_cx: float = None,
    subject_cy: float = None,
    seed: int = None,
) -> dict:
    """
    Convenience: render multiple styles at once.
    Returns {style_name: filtered_image}.
    """
    return {
        name: apply_film_filter(image, name, strength, subject_cx, subject_cy, seed)
        for name in styles
    }


def create_filter_comparison_grid(image: np.ndarray, styles: list = None) -> np.ndarray:
    """
    Render all (or a subset of) styles side-by-side in a labelled grid.
    Returns uint8 RGB.
    """
    if styles is None:
        styles = list_styles()

    filtered = {n: apply_film_filter(image, n) for n in styles}
    panels = {"original": image, **filtered}

    target_h = 400
    labeled = []
    for name, img in panels.items():
        h, w = img.shape[:2]
        scale = target_h / h
        tw = max(1, int(round(w * scale)))
        panel = cv2.resize(img, (tw, target_h), interpolation=cv2.INTER_AREA)
        panel8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        cv2.rectangle(panel8, (0, 0), (tw, 44), (0, 0, 0), -1)
        label = name.replace("_", " ").title()
        cv2.putText(panel8, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        labeled.append(panel8)

    # Stack into rows of up to 4
    cols = 4
    rows = []
    for i in range(0, len(labeled), cols):
        row_panels = labeled[i:i+cols]
        # Pad last row if needed
        while len(row_panels) < cols:
            blank = np.zeros_like(row_panels[0])
            row_panels.append(blank)
        rows.append(np.hstack(row_panels))

    # Ensure all rows have same width
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)

    return np.vstack(padded_rows)
