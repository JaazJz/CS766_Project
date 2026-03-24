import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, '.')

import config
from depth_utils import process_depth, select_focus_interactive

INPUT_FOLDER = "Image_Folder"
OUTPUT_FOLDER = "Batch_Results"
FOCUS_CACHE = "focus_settings.json"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def setup():
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    print(f"✓ Output folder: {OUTPUT_FOLDER}/\n")


def find_images(folder):
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)
    
    images = []
    for file in sorted(os.listdir(folder)):
        if Path(file).suffix in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder, file))
    
    return images


def stage1_select_all_focus(images):

    print("Please click to select focus point for each image")
    
    focus_settings = {}
    total = len(images)
    
    for i, image_path in enumerate(images, 1):
        filename = Path(image_path).name
        print(f"\n[{i}/{total}] {filename}")
        print("-" * 70)
        
        try:
            img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
            H, W = img.shape[:2]
            
            print("  Processing depth...")
            depth_map = process_depth(image_path, W, H, config.DEPTH_MIN, config.DEPTH_MAX)
            
            print("  → Window will popup - CLICK to select focus, press Q")
            focus_distance = select_focus_interactive(img, depth_map)
            
            focus_settings[image_path] = float(focus_distance)
            print(f"  ✓ Saved: {focus_distance:.2f}m")
            
        except KeyboardInterrupt:
            print("\n⚠ Skipping remaining images...")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"  ⚠ Skipping {filename}")
            continue
    
    if focus_settings:
        with open(FOCUS_CACHE, 'w') as f:
            json.dump(focus_settings, f, indent=2)
        
        print(f"Configured: {len(focus_settings)}/{total} images")
        print(f"Settings saved to: {FOCUS_CACHE}")
        print("="*70 + "\n")
    
    return focus_settings


def stage2_batch_process(focus_settings):
    print("You can now leave! Processing will continue automatically.\n")
    
    total = len(focus_settings)
    success_count = 0
    
    for i, (image_path, focus_distance) in enumerate(focus_settings.items(), 1):
        filename = Path(image_path).name
        stem = Path(image_path).stem
        
        print(f"\n[{i}/{total}] {filename} (focus: {focus_distance:.2f}m)")
        print("-" * 70)
        
        comparison_out = os.path.join(OUTPUT_FOLDER, f"{stem}_comparison.png")
        print(f"  → Comparison mode...")
        success_comp = run_main(image_path, 'comparison', focus_distance, comparison_out)
        if success_comp:
            print(f"    ✓ {stem}_comparison.png")
        
        aperture_out = os.path.join(OUTPUT_FOLDER, f"{stem}_apertures.png")
        print(f"  → Aperture mode...")
        success_aper = run_main(image_path, 'aperture', focus_distance, aperture_out)
        if success_aper:
            print(f"    ✓ {stem}_apertures.png")
        
        if success_comp and success_aper:
            success_count += 1
            print(f"  ✓ Complete")
        else:
            print(f"  ⚠ Partial")
    
    print("BATCH PROCESSING COMPLETE!")
    print(f"Successfully processed: {success_count}/{total} images")
    print(f"Results in: {OUTPUT_FOLDER}/")
    
    print("Generated files:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith(('.png', '.jpg')):
            print(f"  • {f}")


def run_main(image_path, mode, focus_distance, output_path):
    cmd = [
        sys.executable,
        "main.py",
        image_path,
        "--focal", str(config.FOCAL_LENGTH),
        "--aperture", str(config.F_NUMBER),
        "--focus", str(focus_distance),
        "--mode", mode,
        "--output", output_path
    ]
    
    try:
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for aperture mode
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Failed with return code {result.returncode}")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                # Print last few lines of error
                for line in error_lines[-3:]:
                    print(f"       {line}")
            return False
        
    except subprocess.TimeoutExpired:
        print(f"Timeout (>5min)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print(f"Camera: {config.FOCAL_LENGTH}mm f/{config.F_NUMBER}")
    print(f"Model:  Depth-Anything V2 ({config.ENCODER})")
    
    setup()
    
    images = find_images(INPUT_FOLDER)
    if not images:
        print(f"No images found in {INPUT_FOLDER}/")
        print(f"Supported: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)
    
    print(f"Found {len(images)} image(s):")
    for img in images:
        print(f"  • {Path(img).name}")
    print()
    
    if os.path.exists(FOCUS_CACHE):
        print(f"Found existing focus settings: {FOCUS_CACHE}")
        response = input("Use existing settings? (y/n): ").strip().lower()
        
        if response == 'y':
            with open(FOCUS_CACHE, 'r') as f:
                focus_settings = json.load(f)
            print(f"Loaded {len(focus_settings)} focus settings\n")
            
            # Go directly to stage 2
            stage2_batch_process(focus_settings)
            return
        else:
            print("Starting fresh focus selection...\n")
    
    focus_settings = stage1_select_all_focus(images)
    
    if not focus_settings:
        print("No images configured. Exiting.")
        sys.exit(1)
    
    response = input("Press ENTER to continue (or Ctrl+C to quit)...")
    
    stage2_batch_process(focus_settings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)