#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict on Mac

import argparse
import numpy as np
from PIL import Image
import config
from depth_utils import process_depth, compute_coc, select_focus_interactive
from renderer import render_dof, create_side_by_side, create_aperture_comparison, create_focus_sweep, save_image

def load_image(path):
    """Load image as float RGB"""
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    print(f"Loaded: {W}x{H}")
    return img, W, H

def main():
    parser = argparse.ArgumentParser(description='Depth-of-Field Simulator')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', default='output.png', help='Output path')
    parser.add_argument('--preset', choices=['portrait', 'landscape', 'macro'], help='Camera preset')
    parser.add_argument('--focal', type=float, help='Focal length (mm)')
    parser.add_argument('--aperture', type=float, help='f-number')
    parser.add_argument('--focus', type=float, help='Focus distance (m), skip interactive')
    parser.add_argument('--mode', choices=['basic', 'comparison', 'aperture', 'sweep'], 
                       default='basic', help='Rendering mode')
    args = parser.parse_args()
    
    # Apply preset
    if args.preset:
        preset = config.PRESETS[args.preset]
        for key, val in preset.items():
            setattr(config, key, val)
        print(f"Applied preset: {args.preset}")
    
    # Override with CLI args
    if args.focal:
        config.FOCAL_LENGTH = args.focal
    if args.aperture:
        config.F_NUMBER = args.aperture
    
    print(f"Settings: {config.FOCAL_LENGTH}mm f/{config.F_NUMBER}")
    
    # Load image
    image, W, H = load_image(args.image)
    depth = process_depth(args.image, W, H, config.DEPTH_MIN, config.DEPTH_MAX)
    
    # Select focus
    if args.focus:
        focus_distance = args.focus
        print(f"Using manual focus: {focus_distance:.2f}m")
    else:
        focus_distance = select_focus_interactive(image, depth)
    print(f"Focus distance: {focus_distance:.2f}m\n")
    
    # Render based on mode
    if args.mode == 'basic':
        # Basic rendering
        coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, focus_distance,
                         config.SENSOR_WIDTH, W, config.MAX_BLUR_PX)
        result = render_dof(image, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
        save_image(result, args.output)
        
    elif args.mode == 'comparison':
        # Before/after comparison
        print("Creating comparison image...")
        coc = compute_coc(depth, config.FOCAL_LENGTH, config.F_NUMBER, focus_distance,
                         config.SENSOR_WIDTH, W, config.MAX_BLUR_PX)
        result = render_dof(image, coc, config.NUM_LAYERS, config.MAX_BLUR_PX)
        comparison = create_side_by_side(image, result, depth)
        save_image(comparison, args.output.replace('.png', '_comparison.png'))
        
    elif args.mode == 'aperture':
        # Aperture comparison grid
        print("Creating aperture comparison (f/1.4 to f/8.0)...")
        grid = create_aperture_comparison(image, depth, focus_distance, 
                                         config.FOCAL_LENGTH, config.SENSOR_WIDTH, 
                                         config.MAX_BLUR_PX)
        save_image(grid, args.output.replace('.png', '_apertures.png'))
        print("\nCreated 2x3 grid with apertures: f/1.4, f/2.0, f/2.8, f/4.0, f/5.6, f/8.0")
        
    elif args.mode == 'sweep':
        # Focus sweep animation
        print("Creating focus sweep animation...")
        frames = create_focus_sweep(image, depth, config.FOCAL_LENGTH, config.F_NUMBER,
                                    config.SENSOR_WIDTH, config.MAX_BLUR_PX, num_frames=10)
        
        # Save as individual frames
        for i, frame in enumerate(frames):
            save_image(frame, f"sweep_frame_{i:02d}.png")
        
        # Try to create GIF
        try:
            import imageio
            imageio.mimsave(args.output.replace('.png', '_sweep.gif'), frames, duration=0.5)
            print(f"Saved animation: {args.output.replace('.png', '_sweep.gif')}")
        except ImportError:
            print("Install imageio for GIF: pip install imageio")
            print(f"Saved 10 frames as: sweep_frame_00.png to sweep_frame_09.png")
    
    print("Done! Try different modes:")
    print(f"  python main.py {args.image} --mode comparison")
    print(f"  python main.py {args.image} --mode aperture")
    print(f"  python main.py {args.image} --mode sweep")

if __name__ == '__main__':
    main()