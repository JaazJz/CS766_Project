# CS766 Project

Automatic photo augmentation from a single image.

## Main components

- `CS766_AutoPhoto_Core/`: depth estimation, subject prioritization, smart crop, and enhancement pipeline
- `CS766_AutoPhoto_Webpage/`: static demo site and asset generation scripts

## Run the pipeline

```bash
python CS766_Project/CS766_AutoPhoto_Core/auto_augment.py Image_Folder --aspect 4:5 --output-dir Auto_Augment_Results
```

## Refresh the webpage demo

```bash
python CS766_Project/CS766_AutoPhoto_Webpage/prepare_webpage_images.py Image_Folder/pic-24.jpg --output-dir CS766_Project/CS766_AutoPhoto_Webpage/webpage_images
python CS766_Project/CS766_AutoPhoto_Webpage/update_html_paths.py webpage_images
```
