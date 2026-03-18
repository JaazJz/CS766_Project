#!/usr/bin/env python3
"""
Webpage Image Preparation Script
Generates all required images for the project webpage from a single input image.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add current directory to path to import project modules
sys.path.insert(0, '.')

import config
from depth_utils import process_depth, compute_coc, select_focus_interactive
from renderer import render_dof, save_image


def create_depth_visualization(depth_map, output_path):
    """Create a colorized depth map visualization"""
    # Normalize depth
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply colormap (TURBO for better visibility)
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved depth visualization: {output_path}")


def create_coc_visualization(coc_map, output_path):
    """Create a visualization of the Circle of Confusion map"""
    # Normalize CoC to 0-1
    coc_vis = (coc_map - coc_map.min()) / (coc_map.max() - coc_map.min() + 1e-8)
    
    # Apply colormap (hot colormap shows blur intensity)
    coc_colored = cv2.applyColorMap((coc_vis * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    coc_colored = cv2.cvtColor(coc_colored, cv2.COLOR_BGR2RGB)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(coc_colored, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved CoC visualization: {output_path}")


def create_pipeline_diagram(image, depth_colored, refocused, output_path):
    """Create a pipeline visualization showing all stages"""
    H, W = image.shape[:2]
    
    # Resize all to same height (300px)
    target_h = 300
    target_w = int(W * target_h / H)
    
    img1 = cv2.resize((image * 255).astype(np.uint8), (target_w, target_h))
    img2 = cv2.resize(depth_colored, (target_w, target_h))
    img3 = cv2.resize((refocused * 255).astype(np.uint8), (target_w, target_h))
    
    # Create canvas with padding
    padding = 40
    arrow_width = 80
    total_width = target_w * 3 + arrow_width * 2 + padding * 2
    total_height = target_h + padding * 2 + 60  # Extra space for labels
    
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place images
    y_offset = padding + 40
    x_positions = [padding, padding + target_w + arrow_width, padding + target_w * 2 + arrow_width * 2]
    
    canvas[y_offset:y_offset+target_h, x_positions[0]:x_positions[0]+target_w] = img1
    canvas[y_offset:y_offset+target_h, x_positions[1]:x_positions[1]+target_w] = img2
    canvas[y_offset:y_offset+target_h, x_positions[2]:x_positions[2]+target_w] = img3
    
    # Add arrows
    arrow_y = y_offset + target_h // 2
    for i in range(2):
        arrow_x = x_positions[i] + target_w + arrow_width // 2
        cv2.arrowedLine(canvas, 
                       (x_positions[i] + target_w + 10, arrow_y),
                       (x_positions[i+1] - 10, arrow_y),
                       (102, 126, 234), 4, tipLength=0.3)
    
    # Add labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = None
    
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    
    labels = ["Input Image", "Depth Map", "Refocused Output"]
    for i, label in enumerate(labels):
        x = x_positions[i] + target_w // 2
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width // 2, 10), label, fill=(102, 126, 234), font=font)
        else:
            cv2.putText(canvas, label, (x - len(label) * 8, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
    
    # Save
    if font:
        pil_canvas.save(output_path, quality=90)
    else:
        cv2.imwrite(output_path, canvas)
    
    print(f"✓ Saved pipeline diagram: {output_path}")


def create_method_images(image, depth, refocused, output_dir):
    """Create detailed method explanation images"""
    H, W = image.shape[:2]
    
    # 1. Depth estimation process
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Create side-by-side
    method_depth = np.hstack([
        (image * 255).astype(np.uint8),
        np.ones((H, 40, 3), dtype=np.uint8) * 255,  # Spacer
        depth_colored
    ])
    cv2.imwrite(os.path.join(output_dir, 'method-depth.jpg'), 
                cv2.cvtColor(method_depth, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved method-depth.jpg")
    
    # 2. Deblurring process (use original as "before" since we don't have Restormer here)
    method_deblur = np.hstack([
        (image * 255).astype(np.uint8),
        np.ones((H, 40, 3), dtype=np.uint8) * 255,
        (image * 255).astype(np.uint8)  # Placeholder
    ])
    cv2.imwrite(os.path.join(output_dir, 'method-deblur.jpg'), 
                cv2.cvtColor(method_deblur, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved method-deblur.jpg")
    
    # 3. Refocusing process
    method_refocus = np.hstack([
        (image * 255).astype(np.uint8),
        np.ones((H, 40, 3), dtype=np.uint8) * 255,
        (refocused * 255).astype(np.uint8)
    ])
    cv2.imwrite(os.path.join(output_dir, 'method-refocus.jpg'), 
                cv2.cvtColor(method_refocus, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved method-refocus.jpg")


def generate_webpage_images(image_path, output_dir, focus_distance=None):
    """Generate all required images for the webpage"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"\nProcessing: {image_path}")
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    print(f"Image size: {W}x{H}")
    
    # Save original image
    original_path = os.path.join(output_dir, "original.jpg")
    Image.fromarray((img * 255).astype(np.uint8)).save(original_path, quality=90)
    print(f"✓ Saved original: {original_path}")
    
    # 1. Generate depth map
    print("\n[1/5] Generating depth map...")
    depth = process_depth(image_path, W, H, config.DEPTH_MIN, config.DEPTH_MAX)
    
    # Save depth visualization
    depth_path = os.path.join(output_dir, "depth.jpg")
    create_depth_visualization(depth, depth_path)
    
    focus_distance = select_focus_interactive(img, depth)

    
    # # 2. Determine focus distance
    # if focus_distance is None:
    #     # Use middle depth as focus
    #     focus_distance = np.median(depth)
    #     print(f"\nAuto-selected focus distance: {focus_distance:.2f}m")
    # else:
    #     print(f"\nUsing manual focus distance: {focus_distance:.2f}m")
    
    # 3. Generate refocused image with default settings
    print("\n[2/5] Generating refocused image...")
    coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, focus_distance,
                     config.SENSOR_WIDTH, W, config.MAX_BLUR_PX)
    refocused = render_dof(img, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
    
    refocused_path = os.path.join(output_dir, "refocused.jpg")
    save_image(refocused, refocused_path)
    
    # 4. Generate CoC visualization
    print("\n[3/5] Generating CoC visualization...")
    coc_path = os.path.join(output_dir, "placeholder-coc.jpg")
    create_coc_visualization(coc, coc_path)
    
    # 5. Generate different apertures
    print("\n[4/5] Generating aperture variations...")
    apertures = [1.0,1.4, 2.8, 5.6, 8.0,11.0]
    
    for f_num in apertures:
        print(f"  Rendering f/{f_num}...")
        coc_aperture = compute_coc(depth, config.FOCAL_LENGTH, f_num, focus_distance,
                                  config.SENSOR_WIDTH, W, config.MAX_BLUR_PX)
        result = render_dof(img, coc_aperture, config.NUM_LAYERS, config.MAX_BLUR_PX)
        
        aperture_path = os.path.join(output_dir, f"placeholder-f{f_num:.1f}.jpg")
        save_image(result, aperture_path)
    
    # 6. Generate method explanation images
    print("\n[5/5] Generating method explanation images...")
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    create_pipeline_diagram(img, depth_colored, refocused, 
                          os.path.join(output_dir, "pipeline-diagram.jpg"))
    create_method_images(img, depth, refocused, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL IMAGES GENERATED!")
    print("="*70)
    print(f"\nImages saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  • original.jpg           - Original input image")
    print("  • depth.jpg              - Depth map visualization")
    print("  • refocused.jpg          - Refocused result")
    print("  • placeholder-coc.jpg    - Circle of Confusion map")
    print("  • placeholder-f1.2.jpg   - Wide aperture (f/1.2)")
    print("  • placeholder-f1.4.jpg   - Wide aperture (f/1.4)")
    print("  • placeholder-f2.8.jpg   - Standard aperture (f/2.8)")
    print("  • placeholder-f5.6.jpg   - Medium aperture (f/5.6)")
    print("  • placeholder-f8.0.jpg   - Narrow aperture (f/8.0)")
    print("  • placeholder-f11.0.jpg   - Narrow aperture (f/11.0)")

    print("  • pipeline-diagram.jpg   - Method overview diagram")
    print("  • method-depth.jpg       - Depth estimation step")
    print("  • method-deblur.jpg      - Deblurring step")
    print("  • method-refocus.jpg     - Refocusing step")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python update_html_paths.py " + output_dir)
    print("2. Open index.html in your browser")
    print("3. Optionally generate focus sweep:")
    print(f"   python main.py {image_path} --mode sweep --focus {focus_distance:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate all webpage images from a single input image'
    )
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output-dir', '-o', default='webpage_images',
                       help='Output directory (default: webpage_images)')
    parser.add_argument('--focus', type=float, 
                       help='Focus distance in meters (auto-detected if not specified)')
    parser.add_argument('--focal', type=float,
                       help=f'Focal length in mm (default: {config.FOCAL_LENGTH})')
    parser.add_argument('--aperture', type=float,
                       help=f'Default f-number (default: {config.F_NUMBER})')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.focal:
        config.FOCAL_LENGTH = args.focal
    if args.aperture:
        config.F_NUMBER = args.aperture
    
    print("="*70)
    print("WEBPAGE IMAGE GENERATOR")
    print("="*70)
    print(f"Camera settings: {config.FOCAL_LENGTH}mm f/{config.F_NUMBER}")
    print(f"Depth range: {config.DEPTH_MIN}m to {config.DEPTH_MAX}m")
    print(f"Model: Depth-Anything V2 ({config.ENCODER})")
    print("="*70)
    
    try:
        generate_webpage_images(args.image, args.output_dir, args.focus)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
