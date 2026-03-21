1. For the CS766 webpage demo, see `CS566_Refocus_Webpage/index.html`.

   Run and update the webpage assets:

   ```bash
   python3.12 prepare_webpage_images.py Test_IMG.jpg -o webpage_images
   python update_html_paths.py webpage_images
   ```

2. For the inherited manual single-image refocus baseline, use the base code:

   ```bash
   python CS766_Project/Refocus_BaseCode/main.py CS766_Project/Refocus_BaseCode/Test_IMG.jpg --mode comparison
   ```

3. For the CS766 extension, use the automatic photo augmentation pipeline:

   ```bash
   python CS766_Project/Refocus_BaseCode/auto_augment.py Image_Folder --aspect 4:5 --output-dir Auto_Augment_Results
   ```

   This adds automatic subject emphasis, adaptive crop, and lightweight style enhancement on top of the original refocus project.
