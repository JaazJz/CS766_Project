import numpy as np
import cv2
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))
import config

# Global model instance
_depth_model = None
_depth_import_error = None
_fallback_notice_shown = False

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception as exc:
    DepthAnythingV2 = None
    _depth_import_error = exc

def get_depth_model():
    global _depth_model
    if _depth_model is None:
        if DepthAnythingV2 is None:
            raise RuntimeError(f"Depth-Anything import failed: {_depth_import_error}")
        print(f"Loading Depth-Anything V2 model ({config.ENCODER})...")
        print(f"Device: {config.DEVICE}")
        model = DepthAnythingV2(**config.MODEL_CONFIGS[config.ENCODER])
        model.load_state_dict(
            torch.load(config.MODEL_CHECKPOINT, map_location='cpu')
        )
        model = model.to(config.DEVICE).eval()
        _depth_model = model
        
        print("Model loaded successfully")
    
    return _depth_model


def estimate_depth_fallback(image_path):
    """Heuristic pseudo-depth when the learned model is unavailable."""
    global _fallback_notice_shown

    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 5.0))
    detail = (detail - detail.min()) / (detail.max() - detail.min() + 1e-8)
    saturation = hsv[:, :, 1]
    saturation = (saturation - saturation.min()) / (saturation.max() - saturation.min() + 1e-8)

    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(width - 1, 1)
    yy = yy / max(height - 1, 1)
    center = np.exp(-(((xx - 0.5) ** 2) / 0.12 + ((yy - 0.48) ** 2) / 0.20))
    center = (center - center.min()) / (center.max() - center.min() + 1e-8)

    brightness = rgb.mean(axis=2)
    brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-8)

    foreground_score = 0.45 * detail + 0.25 * saturation + 0.20 * center + 0.10 * (1.0 - brightness)
    foreground_score = cv2.GaussianBlur(foreground_score.astype(np.float32), (0, 0), 5.0)
    foreground_score = (foreground_score - foreground_score.min()) / (foreground_score.max() - foreground_score.min() + 1e-8)
    pseudo_depth = 1.0 - foreground_score

    if not _fallback_notice_shown:
        print("Warning: Depth-Anything is unavailable; using pseudo-depth fallback.")
        print(f"Reason: {_depth_import_error}")
        _fallback_notice_shown = True

    return pseudo_depth.astype(np.float32)
	
def estimate_depth(image_path):
    """Estimate depth using Depth-Anything V2"""
    try:
        model = get_depth_model()
    except Exception:
        return estimate_depth_fallback(image_path)
    
    # Load image with OpenCV (BGR)
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Infer depth
    with torch.no_grad():
        depth = model.infer_image(raw_img)
    
    return depth

def process_depth(image_path, target_width, target_height, depth_min, depth_max):
    """Complete depth processing pipeline"""
    # Estimate depth
    Z = estimate_depth(image_path).astype(np.float32)
    print(f"Original depth range: [{Z.min():.3f}, {Z.max():.3f}]")
    
    # Normalize to 0-1
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
    
    # Resize to match image
    Z = cv2.resize(Z, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Smooth while preserving edges
    Z = cv2.bilateralFilter(Z, d=9, sigmaColor=0.05, sigmaSpace=5)
    
    # Map to real distances (exponential for realism)
    Z_real = depth_min * np.exp(Z * np.log(depth_max / depth_min))
    print(f"Mapped depth range: [{Z_real.min():.2f}m, {Z_real.max():.2f}m]")
    
    # Clean up
    Z_real = np.clip(Z_real, depth_min, depth_max)
    Z_real = cv2.medianBlur(Z_real, 3)
    
    return Z_real

def compute_coc(depth_map, focal_length, f_number, focus_distance, 
                sensor_width, image_width, max_blur):
    """Calculate Circle of Confusion radius in pixels"""
    pixel_size = sensor_width / image_width
    f_m = focal_length / 1000.0
    aperture_diameter = focal_length / f_number
    
    # CoC = (A * |s - z| * f) / (z * (s - f))
    eps = 1e-6
    numerator = aperture_diameter * np.abs(focus_distance - depth_map) * f_m
    denominator = depth_map * (focus_distance - f_m) + eps
    coc_diameter = numerator / (denominator + eps)
    
    coc_radius = (coc_diameter / pixel_size) / 2.0
    coc_radius = np.clip(coc_radius, 0.0, max_blur)
    coc_radius = np.nan_to_num(coc_radius, nan=0.0, posinf=max_blur)
    
    print(f"CoC range: [{coc_radius.min():.2f}, {coc_radius.max():.2f}] pixels")
    return coc_radius.astype(np.float32)

def select_focus_interactive(image, depth_map):
    """Click to select focus point - Press Q or ESC to confirm"""
    H, W = image.shape[:2]
    selected = {'depth': depth_map[H//2, W//2], 'confirmed': False}
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected['depth'] = depth_map[y, x]
            display = image.copy()
            
            # Draw crosshair
            cv2.drawMarker(display, (x, y), (0, 1, 0), cv2.MARKER_CROSS, 30, 2)
            
            # Big prominent text
            cv2.rectangle(display, (0, 0), (W, 80), (0, 0, 0), -1)  # Black background
            cv2.putText(display, f"Focus: {selected['depth']:.2f}m", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, "Press Q or ESC to CONFIRM", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Select Focus', cv2.cvtColor((display*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"✓ Selected depth: {selected['depth']:.2f}m (press Q or ESC to confirm)")
        
        # Double-click to confirm
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            selected['confirmed'] = True
    
    cv2.namedWindow('Select Focus', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Focus', mouse_callback)
    
    # Initial display with clear instructions
    display = image.copy()
    cv2.rectangle(display, (0, 0), (W, 100), (0, 0, 0), -1)
    cv2.putText(display, "CLICK on object to focus", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(display, "Then press Q or ESC", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imshow('Select Focus', cv2.cvtColor((display*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print("INTERACTIVE FOCUS SELECTION")
    print("1. CLICK on the object you want to focus")
    print("2. Press 'Q' or 'ESC' to confirm and continue")
    
    # Wait for key press or double-click
    while not selected['confirmed']:
        key = cv2.waitKey(100) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # q, Q, or ESC
            break
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Extra wait to ensure window closes on Mac
    
    print(f"\n>>> Confirmed focus distance: {selected['depth']:.2f}m\n")
    return selected['depth']
