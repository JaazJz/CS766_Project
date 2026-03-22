# CS766 Project

Automatic photo augmentation from a single image.

## Main components
- `CS766_AutoPhoto_Core/`: depth estimation, subject prioritization, smart crop, and enhancement pipeline
- `CS766_AutoPhoto_Webpage/`: static demo site and asset generation scripts

## Run the pipeline
```bash
python3.12 CS766_AutoPhoto_Core/auto_augment.py Image_Folder \
    --aspect 4:5 --output-dir Auto_Augment_Results \
    --filter kodak_portra --filter-strength 0.8 --subject-mode auto
```

## Refresh the webpage demo
```bash
python3.12 CS766_AutoPhoto_Webpage/prepare_webpage_images.py Image_Folder/pic-24.jpg \
    --output-dir CS766_AutoPhoto_Webpage/webpage_images \
    --filter-strength 0.8 --subject-mode scene

python3.12 CS766_AutoPhoto_Webpage/update_html_paths.py CS766_AutoPhoto_Webpage/webpage_images
```
## Access the webpage with following link: https://jaazjz.github.io/CS766_Project/
