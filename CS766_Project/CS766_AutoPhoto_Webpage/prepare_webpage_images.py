#!/usr/bin/env python3
"""
Generate webpage assets for the CS766 automatic photo augmentation demo.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CODE_DIR = SCRIPT_DIR.parent / "CS766_AutoPhoto_Core"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASE_CODE_DIR))

import config
from augmentation_utils import (
    apply_style_enhancement,
    compute_crop_box,
    compute_subject_saliency,
    create_overlay,
    create_summary_panel,
    crop_array,
    estimate_focus_distance,
    extract_subject_mask,
    preserve_subject_focus,
)
from depth_utils import compute_coc, process_depth, select_focus_interactive
from renderer import render_dof, save_image


def save_rgb(image, output_path, quality=95):
    image8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image8).save(output_path, quality=quality, optimize=True)
    print(f"Saved: {output_path}")


def colorize_depth(depth_map):
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def colorize_coc(coc_map):
    coc_vis = (coc_map - coc_map.min()) / (coc_map.max() - coc_map.min() + 1e-8)
    coc_colored = cv2.applyColorMap((coc_vis * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return cv2.cvtColor(coc_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def fit_height(image, target_height):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_pipeline_diagram(images, labels, output_path):
    target_height = 320
    gap = 20
    header_height = 64
    panels = [fit_height(img, target_height) for img in images]
    total_width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1) + 40
    canvas = np.ones((target_height + header_height + 40, total_width, 3), dtype=np.uint8) * 248

    x = 20
    y = header_height
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        panel8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        canvas[y:y + panel8.shape[0], x:x + panel8.shape[1]] = panel8
        cv2.putText(canvas, label, (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (52, 72, 102), 2)
        if idx < len(panels) - 1:
            arrow_x0 = x + panel8.shape[1] + 4
            arrow_x1 = arrow_x0 + gap - 8
            arrow_y = y + panel8.shape[0] // 2
            cv2.arrowedLine(canvas, (arrow_x0, arrow_y), (arrow_x1, arrow_y), (210, 110, 48), 3, tipLength=0.22)
        x += panel8.shape[1] + gap

    Image.fromarray(canvas).save(output_path, quality=95, optimize=True)
    print(f"Saved: {output_path}")


def create_focus_sweep_gif(image, depth, output_path, num_frames=10):
    frames = []
    width = image.shape[1]
    focus_distances = np.linspace(depth.min(), depth.max(), num_frames)

    for idx, focus_distance in enumerate(focus_distances, start=1):
        print(f"Generating focus sweep frame {idx}/{num_frames} at {focus_distance:.2f}m")
        coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, focus_distance, config.SENSOR_WIDTH, width, config.MAX_BLUR_PX)
        frame = render_dof(image, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
        frame8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        cv2.putText(frame8, f"Focus {focus_distance:.1f}m", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frames.append(Image.fromarray(frame8))

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=450, loop=0, optimize=True)
    print(f"Saved: {output_path}")


def create_auto_assets(image, depth, output_dir, aspect, focus_distance):
    saliency = compute_subject_saliency(image, depth)
    subject_mask, subject_bbox = extract_subject_mask(saliency)

    if focus_distance is None:
        focus_distance = estimate_focus_distance(depth, subject_mask, saliency)
        print(f"Automatic focus distance: {focus_distance:.2f}m")

    coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, focus_distance, config.SENSOR_WIDTH, image.shape[1], config.MAX_BLUR_PX)
    refocused_raw = render_dof(image, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
    refocused = preserve_subject_focus(image, refocused_raw, subject_mask)

    crop_box = compute_crop_box(image.shape, subject_mask, subject_bbox, aspect)
    auto_original = crop_array(image, crop_box)
    auto_augmented = apply_style_enhancement(crop_array(refocused, crop_box), crop_array(subject_mask, crop_box))
    auto_subject = create_overlay(image, saliency, subject_bbox=subject_bbox, crop_box=crop_box)
    auto_summary = create_summary_panel(auto_original, crop_array(auto_subject, crop_box), auto_augmented)

    save_rgb(auto_original, output_dir / "auto_original.jpg")
    save_rgb(auto_augmented, output_dir / "auto_augmented.jpg")
    save_rgb(auto_subject, output_dir / "auto_subject.jpg")
    save_rgb(auto_summary, output_dir / "auto_summary.jpg")
    save_rgb(auto_summary, output_dir / "method-augment.jpg")

    return {
        "focus_distance": float(focus_distance),
        "refocused": refocused,
        "saliency_overlay": auto_subject,
        "auto_summary": auto_summary,
        "auto_original": auto_original,
        "auto_augmented": auto_augmented,
        "crop_box": crop_box,
    }


def create_method_images(image, depth_color, refocused, auto_summary, output_dir):
    spacer = np.ones((image.shape[0], max(24, image.shape[1] // 30), 3), dtype=np.float32)

    method_depth = np.hstack([image, spacer, depth_color])
    save_rgb(method_depth, output_dir / "method-depth.jpg")

    method_refocus = np.hstack([image, spacer, refocused])
    save_rgb(method_refocus, output_dir / "method-refocus.jpg")

    save_rgb(auto_summary, output_dir / "method-augment.jpg")


def generate_webpage_assets(image_path, output_dir, focus, interactive_focus, aspect):
    output_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    print(f"Loaded {image_path.name}: {width}x{height}")

    save_rgb(image, output_dir / "original.jpg")

    depth = process_depth(str(image_path), width, height, config.DEPTH_MIN, config.DEPTH_MAX)
    depth_color = colorize_depth(depth)
    save_rgb(depth_color, output_dir / "depth.jpg")

    focus_distance = focus
    if interactive_focus and focus_distance is None:
        focus_distance = select_focus_interactive(image, depth)

    auto_assets = create_auto_assets(image, depth, output_dir, aspect, focus_distance)

    refocused = auto_assets["refocused"]
    save_rgb(refocused, output_dir / "refocused.jpg")

    coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, auto_assets["focus_distance"], config.SENSOR_WIDTH, width, config.MAX_BLUR_PX)
    save_rgb(colorize_coc(coc), output_dir / "coc.jpg")

    for aperture in [1.0, 1.4, 2.8, 5.6, 8.0, 11.0]:
        print(f"Generating aperture sample f/{aperture}")
        aperture_coc = compute_coc(depth, config.FOCAL_LENGTH, aperture, auto_assets["focus_distance"], config.SENSOR_WIDTH, width, config.MAX_BLUR_PX)
        aperture_result = render_dof(image, aperture_coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
        save_rgb(aperture_result, output_dir / f"f{aperture:.1f}.jpg")

    create_focus_sweep_gif(image, depth, output_dir / "focus-sweep.gif")

    build_pipeline_diagram(
        images=[
            image,
            depth_color,
            auto_assets["saliency_overlay"],
            crop_array(auto_assets["saliency_overlay"], auto_assets["crop_box"]),
            auto_assets["auto_augmented"],
        ],
        labels=["Input", "Depth", "Subject Prior", "Adaptive Crop", "Final Augment"],
        output_path=output_dir / "pipeline-diagram.jpg",
    )

    create_method_images(image, depth_color, refocused, auto_assets["auto_summary"], output_dir)
    print(f"Webpage assets saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate webpage images for the CS766 project demo.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output-dir", "-o", default="webpage_images", help="Output directory")
    parser.add_argument("--focus", type=float, help="Manually set focus distance in meters")
    parser.add_argument("--interactive-focus", action="store_true", help="Select focus interactively instead of automatic estimation")
    parser.add_argument("--aspect", default="4:5", help="Target crop aspect ratio for automatic augmentation")
    parser.add_argument("--focal", type=float, help="Override focal length in mm")
    parser.add_argument("--aperture", type=float, help="Override f-number")
    args = parser.parse_args()

    if args.focal:
        config.FOCAL_LENGTH = args.focal
    if args.aperture:
        config.F_NUMBER = args.aperture

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input not found: {args.image}")

    print("=" * 70)
    print("CS766 WEBPAGE ASSET GENERATOR")
    print("=" * 70)
    print(f"Camera settings: {config.FOCAL_LENGTH}mm f/{config.F_NUMBER}")
    print(f"Depth range: {config.DEPTH_MIN}m to {config.DEPTH_MAX}m")
    print("=" * 70)

    generate_webpage_assets(
        image_path=image_path,
        output_dir=Path(args.output_dir),
        focus=args.focus,
        interactive_focus=args.interactive_focus,
        aspect=args.aspect,
    )


if __name__ == "__main__":
    main()
