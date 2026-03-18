#!/usr/bin/env python3
"""
Automatically update index.html with correct image paths
"""

import re
import sys
from pathlib import Path


def update_html_images(html_path, image_dir):
    import re

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1) 把占位符文件名改成目标文件名（只处理还没替过的情况）
    placeholder_map = {
        'placeholder-original.jpg': 'original.jpg',
        'placeholder-depth.jpg': 'depth.jpg',
        'placeholder-refocused.jpg': 'refocused.jpg',
        'placeholder-coc.jpg': 'placeholder-coc.jpg',
        'placeholder-f1.4.jpg': 'placeholder-f1.4.jpg',
        'placeholder-f2.8.jpg': 'placeholder-f2.8.jpg',
        'placeholder-f5.6.jpg': 'placeholder-f5.6.jpg',
        'placeholder-f8.0.jpg': 'placeholder-f8.0.jpg',
        'placeholder-focus-sweep.gif': 'placeholder-focus-sweep.gif',
        'pipeline-diagram.jpg': 'pipeline-diagram.jpg',
        'method-depth.jpg': 'method-depth.jpg',
        'method-deblur.jpg': 'method-deblur.jpg',
        'method-refocus.jpg': 'method-refocus.jpg',
    }

    updated = False
    for old, new_base in placeholder_map.items():
        if old in content:
            content = content.replace(old, new_base)
            updated = True
            print(f"✓ Replaced placeholder name: {old} → {new_base}")

    # 2) 按“文件名”重定向到新目录（可多次运行 & 可切换目录）
    #    只要是以这些文件名结尾的路径，统统替换成 image_dir/文件名
    target_basenames = sorted(set(placeholder_map.values()))
    # 组装一个捕获文件名的正则，如 (...)?(original\.jpg|depth\.jpg|...)
    name_alt = "|".join(re.escape(b) for b in target_basenames)
    # 匹配「任意目录/文件名」或「仅文件名」：比如 img/foo/original.jpg 或 original.jpg
    # 我们只捕获最终文件名，前面的目录（若有）会被丢弃
    pattern = re.compile(r'(?:[\w\-.~/]+/)?(' + name_alt + r')')

    def _retarget(m):
        basename = m.group(1)  # 例如 original.jpg
        return f'{image_dir}/{basename}'

    new_content = pattern.sub(_retarget, content)
    if new_content != content:
        content = new_content
        updated = True
        print(f"✓ Retargeted all image paths to: {image_dir}/<basename>")

    if not updated:
        print("ℹ Nothing to update (paths already correct?)")
        return False

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ Updated: {html_path}")
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
    required_images = ['original.jpg', 'depth.jpg', 'refocused.jpg', 
                      'placeholder-coc.jpg', 'placeholder-f1.4.jpg', 'placeholder-f2.8.jpg', 
                      'placeholder-f5.6.jpg', 'placeholder-f8.0.jpg',
                      'pipeline-diagram.jpg', 'method-depth.jpg', 
                      'method-deblur.jpg', 'method-refocus.jpg']
    
    missing = []
    for img in required_images:
        if not Path(image_dir, img).exists():
            missing.append(img)
    
    if missing:
        print(f"⚠ Warning: Some images are missing from {image_dir}/:")
        for img in missing:
            print(f"  • {img}")
        print("\nMake sure to run prepare_webpage_images.py to generate all images.")
        print("Continuing anyway...")
    
    # Update HTML
    print(f"\nUpdating {html_path}...")
    print("-" * 50)
    
    if update_html_images(html_path, image_dir):
        print("-" * 50)
        print("\n✅ SUCCESS! Your webpage is ready!")
        print(f"\nNext steps:")
        print(f"1. Open {html_path} in your web browser")
        print(f"2. Use the interactive sliders to compare images")
        print(f"3. Check all 5 tabs work correctly")
        print(f"4. Deploy to GitHub Pages or Netlify when ready")
    else:
        print("\n✓ HTML file is already configured")


if __name__ == '__main__':
    main()
