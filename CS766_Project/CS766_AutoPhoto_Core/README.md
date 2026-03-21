# CS766 Auto Photo Core

This module runs the automatic photo augmentation pipeline:

- depth estimation with a robust fallback path
- automatic subject scoring
- depth-aware focus rendering
- adaptive crop selection
- lightweight style enhancement

Single image:

```bash
python CS766_Project/CS766_AutoPhoto_Core/auto_augment.py Image_Folder/pic-24.jpg --output-dir Auto_Augment_Results
```

Folder batch:

```bash
python CS766_Project/CS766_AutoPhoto_Core/auto_augment.py Image_Folder --aspect 4:5 --output-dir Auto_Augment_Results
```

Outputs per image:

- `*_auto_augmented.png`
- `*_subject_overlay.png`
- `*_summary.png`
- `*_metadata.json`

If `Depth-Anything` is unavailable because of a local `torch/torchvision` mismatch, the pipeline automatically falls back to a lightweight pseudo-depth prior so the demo can still run end-to-end.
