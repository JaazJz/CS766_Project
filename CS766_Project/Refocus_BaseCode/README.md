# CS766_Project
Automatic Photo Augmentation Built on Single-Image Refocusing

Use https://github.com/DepthAnything/Depth-Anything-V2 to download the checkpoint and place it under the repository `checkpoints/` folder.

## Manual refocus

```bash
python CS766_Project/Refocus_BaseCode/main.py CS766_Project/Refocus_BaseCode/Test_IMG.jpg --mode comparison
```

## Automatic photo augmentation

The automatic pipeline adds three proposal-aligned stages on top of the existing refocus code:

- automatic subject emphasis with a depth + saliency heuristic
- adaptive composition adjustment with smart cropping
- lightweight style enhancement for tone, color, and local detail

Run on a single image:

```bash
python CS766_Project/Refocus_BaseCode/auto_augment.py Image_Folder/pic-20.jpg --output-dir Auto_Augment_Results
```

Run on a folder:

```bash
python CS766_Project/Refocus_BaseCode/auto_augment.py Image_Folder --aspect 4:5 --output-dir Auto_Augment_Results
```

Each image produces:

- `*_auto_augmented.png`: final augmented result
- `*_subject_overlay.png`: saliency and crop visualization
- `*_summary.png`: side-by-side summary for reports or demos
- `*_metadata.json`: estimated focus distance, subject box, and crop box

If `Depth-Anything` cannot be imported because of a local `torch/torchvision` mismatch, the pipeline falls back to a lightweight pseudo-depth heuristic so the augmentation demo can still run end-to-end.
