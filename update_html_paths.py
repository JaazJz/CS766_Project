#!/usr/bin/env python3
"""Update index.html to point at a chosen image directory."""

import re
import sys
from pathlib import Path


IMAGE_NAMES = [
    "original.jpg",
    "depth.jpg",
    "refocused.jpg",
    "coc.jpg",
    "f1.0.jpg",
    "f1.4.jpg",
    "f2.8.jpg",
    "f5.6.jpg",
    "f8.0.jpg",
    "f11.0.jpg",
    "focus-sweep.gif",
    "pipeline-diagram.jpg",
    "method-depth.jpg",
    "method-refocus.jpg",
    "method-augment.jpg",
    "auto_original.jpg",
    "auto_augmented.jpg",
    "auto_subject.jpg",
    "auto_summary.jpg",
    # Film filter variants
    "filter_kodak_portra.jpg",
    "filter_fuji_velvia.jpg",
    "filter_fuji_pro400h.jpg",
    "filter_cinestill_800t.jpg",
    "filter_ilford_hp5.jpg",
    "filter_kodak_trix.jpg",
    "filter_teal_orange.jpg",
    "filter_vintage_faded.jpg",
]


def update_html_images(html_path, image_dir):
    content = html_path.read_text(encoding="utf-8")
    update_count = 0

    for image_name in IMAGE_NAMES:
        new_path = f"{image_dir}/{image_name}"
        pattern = r'src="(?:[^"]*[\\/])?' + re.escape(image_name) + r'"'
        replacement = f'src="{new_path}"'
        content, count = re.subn(pattern, replacement, content)
        if count:
            update_count += count
            print(f"Updated {count} occurrence(s) of {image_name}")

    html_path.write_text(content, encoding="utf-8")
    return update_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_html_paths.py <image_directory>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    html_path = Path("index.html")

    if not html_path.exists():
        raise FileNotFoundError("index.html not found in the current directory")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    missing = [name for name in IMAGE_NAMES if not (image_dir / name).exists()]
    if missing:
        print("Warning: some assets are missing:")
        for name in missing:
            print(f"  - {name}")

    updated = update_html_images(html_path, image_dir.as_posix())
    print(f"Updated {updated} image references in {html_path}")


if __name__ == "__main__":
    main()
