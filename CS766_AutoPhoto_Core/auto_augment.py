#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import config
from augmentation_utils import (
    ASPECT_RATIOS,
    apply_style_enhancement,
    create_augmented_image,
    compute_crop_box,
    compute_subject_saliency,
    create_overlay,
    create_summary_panel,
    crop_array,
    estimate_focus_distance,
    extract_subject_mask,
    metadata_to_json,
    preserve_subject_focus,
)
from depth_utils import compute_coc, process_depth
from renderer import render_dof, save_image
from filter_utils import apply_film_filter, list_styles


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

# Styles that get saved as separate outputs when --filter all is used
_ALL_STYLES = list_styles()


def load_image(path):
    image = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    print(f"Loaded {Path(path).name}: {width}x{height}")
    return image, width, height


def resolve_images(input_path):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if path.is_file():
        return [path]

    images = [c for c in sorted(path.iterdir()) if c.suffix in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return images


def run_auto_augment(
    image_path,
    output_dir,
    aspect,
    skip_refocus,
    skip_enhance,
    filter_style,
    filter_strength,
    filter_seed,
    subject_mode="auto",
):
    image, width, height = load_image(str(image_path))
    depth = process_depth(str(image_path), width, height, config.DEPTH_MIN, config.DEPTH_MAX)

    saliency = compute_subject_saliency(image, depth, subject_mode=subject_mode)
    subject_mask, subject_bbox = extract_subject_mask(saliency)
    focus_distance = estimate_focus_distance(depth, subject_mask, saliency)
    print(f"Auto focus distance: {focus_distance:.2f}m")

    # ── Refocus ──────────────────────────────────────────────────────────────
    if skip_refocus:
        emphasized = image
    else:
        coc = compute_coc(
            depth,
            config.FOCAL_LENGTH,
            config.F_NUMBER,
            focus_distance,
            config.SENSOR_WIDTH,
            width,
            config.MAX_BLUR_PX,
        )
        refocused = render_dof(image, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
        emphasized = preserve_subject_focus(image, refocused, subject_mask)

    # ── Crop ─────────────────────────────────────────────────────────────────
    crop_box = compute_crop_box(image.shape, subject_mask, subject_bbox, aspect)
    cropped_result   = crop_array(emphasized,    crop_box)
    cropped_original = crop_array(image,         crop_box)
    cropped_subject  = crop_array(subject_mask,  crop_box)

    # ── Tone/detail enhancement ───────────────────────────────────────────────
    if skip_enhance:
        base_image = cropped_result
    else:
        base_image = create_augmented_image(cropped_original, cropped_result, cropped_subject)

    # ── Subject centre (used for vignette anchoring in filters) ───────────────
    mask_ys, mask_xs = np.where(cropped_subject > 0.15)
    if len(mask_xs):
        subj_cx = float(mask_xs.mean())
        subj_cy = float(mask_ys.mean())
    else:
        subj_cx = subj_cy = None

    # ── Film filter ───────────────────────────────────────────────────────────
    apply_all = (filter_style == "all")
    styles_to_run = _ALL_STYLES if apply_all else ([] if filter_style == "none" else [filter_style])

    # Primary output: no filter (or single named filter)
    if filter_style == "none":
        final_image = base_image
    elif apply_all:
        # Primary output is unfiltered; individual style outputs are saved below
        final_image = base_image
    else:
        print(f"Applying film filter: {filter_style} (strength={filter_strength})")
        final_image = apply_film_filter(
            base_image,
            filter_style,
            strength=filter_strength,
            subject_cx=subj_cx,
            subject_cy=subj_cy,
            seed=filter_seed,
        )

    # ── Diagnostics overlay & summary ─────────────────────────────────────────
    overlay = create_overlay(image, saliency, subject_bbox=subject_bbox, crop_box=crop_box)
    summary = create_summary_panel(image, overlay, final_image)

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path    = output_dir / f"{stem}_auto_augmented.png"
    overlay_path  = output_dir / f"{stem}_subject_overlay.png"
    summary_path  = output_dir / f"{stem}_summary.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    save_image(final_image, str(final_path))
    save_image(overlay,     str(overlay_path))
    save_image(summary,     str(summary_path))

    # When --filter all: save every style as a separate file
    filter_outputs = {}
    if apply_all:
        print(f"Saving all {len(_ALL_STYLES)} film filter variants...")
        for style in _ALL_STYLES:
            filtered = apply_film_filter(
                base_image,
                style,
                strength=filter_strength,
                subject_cx=subj_cx,
                subject_cy=subj_cy,
                seed=filter_seed,
            )
            style_path = output_dir / f"{stem}_filter_{style}.png"
            save_image(filtered, str(style_path))
            filter_outputs[style] = str(style_path)
            print(f"  Saved: {style_path.name}")

    metadata = {
        "image": str(image_path),
        "focus_distance_m": round(focus_distance, 4),
        "aspect_ratio": aspect,
        "skip_refocus": bool(skip_refocus),
        "skip_enhance": bool(skip_enhance),
        "filter_style": filter_style,
        "filter_strength": filter_strength,
        "subject_bbox_xywh": [int(v) for v in subject_bbox],
        "crop_box_xywh": [int(v) for v in crop_box],
        "outputs": {
            "final":   str(final_path),
            "overlay": str(overlay_path),
            "summary": str(summary_path),
            **filter_outputs,
        },
    }
    metadata_to_json(metadata_path, metadata)
    print(f"Saved metadata: {metadata_path}")


def main():
    available_styles = list_styles()

    parser = argparse.ArgumentParser(
        description="Automatic photo augmentation with subject emphasis, smart crop, and film filters."
    )
    parser.add_argument("input",      help="Input image path or folder")
    parser.add_argument("--output-dir", "-o", default="Auto_Augment_Results",
                        help="Where to save augmented outputs")
    parser.add_argument("--aspect",   choices=tuple(ASPECT_RATIOS.keys()), default="4:5",
                        help="Target crop aspect ratio")
    parser.add_argument("--preset",   choices=["portrait", "landscape", "macro"],
                        help="Camera preset")
    parser.add_argument("--focal",    type=float, help="Override focal length (mm)")
    parser.add_argument("--aperture", type=float, help="Override f-number")
    parser.add_argument(
        "--subject-mode",
        choices=["auto", "people", "scene"],
        default="auto",
        help=(
            "Subject detection mode: "
            "'auto' = decide automatically based on people prominence; "
            "'people' = always treat detected people as the subject; "
            "'scene' = always treat the environment as the subject (ignores people detector). "
            "(default: auto)"
        ),
    )
    parser.add_argument("--skip-refocus", action="store_true",
                        help="Disable depth-based refocus; only crop/enhance")
    parser.add_argument("--skip-enhance", action="store_true",
                        help="Disable tone/detail enhancement")

    # ── Film filter args ──────────────────────────────────────────────────────
    parser.add_argument(
        "--filter",
        dest="filter_style",
        choices=available_styles + ["none", "all"],
        default="none",
        metavar="STYLE",
        help=(
            "Film/camera style filter to apply after augmentation. "
            f"Choices: none, all, {', '.join(available_styles)}. "
            "Use 'all' to save every style as a separate file."
        ),
    )
    parser.add_argument(
        "--filter-strength",
        type=float,
        default=1.0,
        metavar="S",
        help="Filter intensity: 0.0 = no effect, 1.0 = full (default: 1.0)",
    )
    parser.add_argument(
        "--filter-seed",
        type=int,
        default=None,
        metavar="N",
        help="RNG seed for reproducible film grain (default: random each run)",
    )

    args = parser.parse_args()

    if args.preset:
        preset = config.PRESETS[args.preset]
        for key, value in preset.items():
            setattr(config, key, value)
        print(f"Applied preset: {args.preset}")

    if args.focal:
        config.FOCAL_LENGTH = args.focal
    if args.aperture:
        config.F_NUMBER = args.aperture

    print(f"Settings: {config.FOCAL_LENGTH}mm f/{config.F_NUMBER}, aspect {args.aspect}")
    if args.filter_style != "none":
        print(f"Film filter: {args.filter_style}  strength={args.filter_strength}")
    print(f"Available filters: {', '.join(available_styles)}\n")

    images     = resolve_images(args.input)
    output_dir = Path(args.output_dir)

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] Processing {image_path.name}")
        run_auto_augment(
            image_path=image_path,
            output_dir=output_dir,
            aspect=args.aspect,
            skip_refocus=args.skip_refocus,
            skip_enhance=args.skip_enhance,
            filter_style=args.filter_style,
            filter_strength=args.filter_strength,
            filter_seed=args.filter_seed,
            subject_mode=args.subject_mode,
        )

    print("\nAutomatic augmentation complete.")


if __name__ == "__main__":
    main()
