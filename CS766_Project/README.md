1. For Webpage, see CS566_Refocus_Webpage/index.html
   
   Run and Update the index with new image in terminal:
   
   python3.12 prepare_webpage_images.py Test_IMG.jpg -o webpage_images
   
   python update_html_paths.py webpage_images
3. To Run the Refocus, use https://github.com/DepthAnything/Depth-Anything-V2 to download CheckPoint for Depth-Anything-V2-Large and put in the Refocus_BaseCode folder.
   
   python3.12 main.py IMAGE_NAME --edge-softness 7 --focal 50 --aperture 1.8
   
   You are free to set your prefered edge-softness/focal/apeture
