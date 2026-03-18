#!/usr/bin/env python3
"""
Automatically update index.html with correct image paths
"""

import re
import sys
from pathlib import Path


def update_html_images(html_path, image_dir):
    """Update HTML file to use images from specified directory"""
    
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple and direct: replace ALL image src paths
    # Pattern matches src="anything" and replaces with new path
    replacements = {
        # Main images
        'original.jpg': f'{image_dir}/original.jpg',
        'depth.jpg': f'{image_dir}/depth.jpg',
        'refocused.jpg': f'{image_dir}/refocused.jpg',
        'coc.jpg': f'{image_dir}/coc.jpg',
        
        # Aperture images
        'f1.0.jpg': f'{image_dir}/f1.0.jpg',
        'f1.4.jpg': f'{image_dir}/f1.4.jpg',
        'f2.8.jpg': f'{image_dir}/f2.8.jpg',
        'f5.6.jpg': f'{image_dir}/f5.6.jpg',
        'f8.0.jpg': f'{image_dir}/f8.0.jpg',
        'f11.0.jpg': f'{image_dir}/f11.0.jpg',
        
        # Method images
        'pipeline-diagram.jpg': f'{image_dir}/pipeline-diagram.jpg',
        'method-depth.jpg': f'{image_dir}/method-depth.jpg',
        'method-deblur.jpg': f'{image_dir}/method-deblur.jpg',
        'method-refocus.jpg': f'{image_dir}/method-refocus.jpg',
        
        # Animation
        'focus-sweep.gif': f'{image_dir}/focus-sweep.gif',
        
        # Old placeholder names (for backward compatibility)
        'placeholder-coc.jpg': f'{image_dir}/coc.jpg',
        'placeholder-f1.0.jpg': f'{image_dir}/f1.0.jpg',
        'placeholder-f1.4.jpg': f'{image_dir}/f1.4.jpg',
        'placeholder-f2.8.jpg': f'{image_dir}/f2.8.jpg',
        'placeholder-f5.6.jpg': f'{image_dir}/f5.6.jpg',
        'placeholder-f8.0.jpg': f'{image_dir}/f8.0.jpg',
        'placeholder-f11.0.jpg': f'{image_dir}/f11.0.jpg',
        'placeholder-focus-sweep.gif': f'{image_dir}/focus-sweep.gif',
        'placeholder-pipeline-diagram.jpg': f'{image_dir}/pipeline-diagram.jpg',
        'placeholder-method-depth.jpg': f'{image_dir}/method-depth.jpg',
        'placeholder-method-deblur.jpg': f'{image_dir}/method-deblur.jpg',
        'placeholder-method-refocus.jpg': f'{image_dir}/method-refocus.jpg',
    }
    
    updated_count = 0
    
    # Replace each filename (with any path before it)
    for old_name, new_path in replacements.items():
        # Match: src="anything/filename" or src="filename"
        pattern = r'src="[^"]*?' + re.escape(old_name) + r'"'
        replacement = f'src="{new_path}"'
        
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            updated_count += count
            print(f"✓ Updated {count} occurrence(s) of: {old_name} → {new_path}")

    if updated_count == 0:
        print("ℹ No image paths found to update")
        return False

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ Updated {html_path} ({updated_count} total updates)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_html_paths.py <image_directory>")
        print("\nExample:")
        print("  python update_html_paths.py webpage_images")
        print("\nThis will update index.html to use images from the specified directory")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    html_path = 'index.html'
    
    # Check if HTML file exists
    if not Path(html_path).exists():
        print(f"❌ Error: {html_path} not found!")
        print("Make sure you're running this in the same directory as index.html")
        sys.exit(1)
    
    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"❌ Error: Directory '{image_dir}' not found!")
        print("\nDid you run prepare_webpage_images.py first?")
        print("Example: python prepare_webpage_images.py your_image.jpg -o webpage_images")
        sys.exit(1)
    
    # Check if images exist
    required_images = [
        'original.jpg', 'depth.jpg', 'refocused.jpg', 'coc.jpg',
        'f1.0.jpg', 'f1.4.jpg', 'f2.8.jpg', 'f5.6.jpg', 'f8.0.jpg', 'f11.0.jpg',
        'focus-sweep.gif',
        'pipeline-diagram.jpg', 'method-depth.jpg', 
        'method-deblur.jpg', 'method-refocus.jpg'
    ]
    
    missing = []
    existing = []
    for img in required_images:
        if Path(image_dir, img).exists():
            existing.append(img)
        else:
            missing.append(img)
    
    if existing:
        print(f"\n✅ Found {len(existing)} images in {image_dir}/:")
        for img in existing[:5]:  # Show first 5
            print(f"  • {img}")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")
    
    if missing:
        print(f"\n⚠ Warning: {len(missing)} images missing from {image_dir}/:")
        for img in missing:
            print(f"  • {img}")
        print("\nRun prepare_webpage_images.py to generate missing images")
    
    # Update HTML
    print(f"\nUpdating {html_path}...")
    print("-" * 50)
    
    if update_html_images(html_path, image_dir):
        print("-" * 50)
        print("\n✅ SUCCESS! Your webpage is ready!")
        print(f"\nNext steps:")
        print(f"1. Open {html_path} in your web browser")
        print(f"2. All images should now load from {image_dir}/")
        print(f"3. Use the interactive sliders to compare images")
    else:
        print("\n⚠ No updates made - check if paths are already correct")


if __name__ == '__main__':
    main()
