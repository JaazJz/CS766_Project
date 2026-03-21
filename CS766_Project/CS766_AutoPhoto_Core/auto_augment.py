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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


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

    images = [candidate for candidate in sorted(path.iterdir()) if candidate.suffix in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return images


def run_auto_augment(image_path, output_dir, aspect, skip_refocus, skip_enhance):
    image, width, height = load_image(str(image_path))
    depth = process_depth(str(image_path), width, height, config.DEPTH_MIN, config.DEPTH_MAX)

    saliency = compute_subject_saliency(image, depth)
    subject_mask, subject_bbox = extract_subject_mask(saliency)
    focus_distance = estimate_focus_distance(depth, subject_mask, saliency)
    print(f"Auto focus distance: {focus_distance:.2f}m")

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

    crop_box = compute_crop_box(image.shape, subject_mask, subject_bbox, aspect)
    cropped_result = crop_array(emphasized, crop_box)
    cropped_original = crop_array(image, crop_box)
    cropped_subject = crop_array(subject_mask, crop_box)

    if skip_enhance:
        final_image = cropped_result
    else:
        final_image = apply_style_enhancement(cropped_result, cropped_subject)

    overlay = create_overlay(image, saliency, subject_bbox=subject_bbox, crop_box=crop_box)
    summary = create_summary_panel(cropped_original, crop_array(overlay, crop_box), final_image)

    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{stem}_auto_augmented.png"
    overlay_path = output_dir / f"{stem}_subject_overlay.png"
    summary_path = output_dir / f"{stem}_summary.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    save_image(final_image, str(final_path))
    save_image(overlay, str(overlay_path))
    save_image(summary, str(summary_path))

    metadata = {
        "image": str(image_path),
        "focus_distance_m": round(focus_distance, 4),
        "aspect_ratio": aspect,
        "skip_refocus": bool(skip_refocus),
        "skip_enhance": bool(skip_enhance),
        "subject_bbox_xywh": [int(v) for v in subject_bbox],
        "crop_box_xywh": [int(v) for v in crop_box],
        "outputs": {
            "final": str(final_path),
            "overlay": str(overlay_path),
            "summary": str(summary_path),
        },
    }
    metadata_to_json(metadata_path, metadata)
    print(f"Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Automatic photo augmentation with subject emphasis and smart crop.")
    parser.add_argument("input", help="Input image path or folder")
    parser.add_argument("--output-dir", "-o", default="Auto_Augment_Results", help="Where to save augmented outputs")
    parser.add_argument("--aspect", choices=tuple(ASPECT_RATIOS.keys()), default="4:5", help="Target crop aspect ratio")
    parser.add_argument("--preset", choices=["portrait", "landscape", "macro"], help="Camera preset")
    parser.add_argument("--focal", type=float, help="Override focal length (mm)")
    parser.add_argument("--aperture", type=float, help="Override f-number")
    parser.add_argument("--skip-refocus", action="store_true", help="Disable depth-based refocus and only crop/enhance")
    parser.add_argument("--skip-enhance", action="store_true", help="Disable tone/detail enhancement")
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
    images = resolve_images(args.input)
    output_dir = Path(args.output_dir)

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] Processing {image_path.name}")
        run_auto_augment(
            image_path=image_path,
            output_dir=output_dir,
            aspect=args.aspect,
            skip_refocus=args.skip_refocus,
            skip_enhance=args.skip_enhance,
        )

    print("\nAutomatic augmentation complete.")


if __name__ == "__main__":
    main()
