import torch
FOCAL_LENGTH = 50.0      # mm
F_NUMBER = 2.8           # f-stop
SENSOR_WIDTH = 36.0      # mm (full-frame)
MAX_BLUR_PX = 40         # pixels

DEPTH_MIN = 0.5          # meters
DEPTH_MAX = 100.0        # meters

NUM_LAYERS = 20          # more = smoother but slower

# Depth model
# MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"
# MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
# MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"
# MODEL_NAME = "LiheYoung/depth-anything-large-hf"
# MODEL_NAME = 'vitl'  # 'vits', 'vitb', 'vitl', 'vitg'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
ENCODER = 'vitl'
MODEL_CHECKPOINT = f'checkpoints/depth_anything_v2_{ENCODER}.pth'

PRESETS = {
    'portrait': {'FOCAL_LENGTH': 85.0, 'F_NUMBER': 1.8, 'MAX_BLUR_PX': 50},
    'landscape': {'FOCAL_LENGTH': 35.0, 'F_NUMBER': 8.0, 'MAX_BLUR_PX': 20},
    'macro': {'FOCAL_LENGTH': 100.0, 'F_NUMBER': 2.8, 'MAX_BLUR_PX': 60},
}