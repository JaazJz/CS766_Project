import numpy as np
import cv2

def render_dof(image, coc_map, num_layers=40, max_radius=40):
    H, W = image.shape[:2]
    
    blur_radii = np.linspace(0, max_radius, num_layers)
    blur_layers = []
    
    print(f"Generating {num_layers} blur layers...")
    for i, radius in enumerate(blur_radii):
        if radius < 0.5:
            blur_layers.append(image.copy())
        else:
            sigma = radius / 2.0
            blurred = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
            blur_layers.append(blurred)
        print(f"  {i+1}/{num_layers}", end='\r')
    print()
    
    layer_indices = np.clip((coc_map / max_radius) * (num_layers - 1), 0, num_layers - 1)
    idx_low = np.floor(layer_indices).astype(np.int32)
    idx_high = np.ceil(layer_indices).astype(np.int32)
    alpha = (layer_indices - idx_low).astype(np.float32)
    
    print("Blending...")
    result = np.zeros_like(image)
    for i in range(num_layers):
        mask_low = (idx_low == i).astype(np.float32)
        mask_high = (idx_high == i).astype(np.float32)
        weight = mask_low * (1 - alpha) + mask_high * alpha
        if weight.max() > 0:
            result += blur_layers[i] * weight[:, :, np.newaxis]
    
    return np.clip(result, 0.0, 1.0)

def create_side_by_side(original, result, depth_map=None):
    H, W = original.shape[:2]
    
    if depth_map is not None:
        depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()))
        depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        combined = np.hstack([original, result, depth_colored])
        
        combined = (combined * 255).astype(np.uint8)
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "With DOF", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return combined / 255.0
    else:
        combined = np.hstack([original, result])
        combined = (combined * 255).astype(np.uint8)
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "With DOF", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return combined / 255.0

def create_focus_sweep(image, depth_map, focal_length, f_number, 
                       sensor_width, max_blur, num_frames=10):
    from depth_utils import compute_coc
    
    W = image.shape[1]
    depths = np.linspace(depth_map.min(), depth_map.max(), num_frames)
    
    frames = []
    for i, focus_dist in enumerate(depths):
        print(f"Frame {i+1}/{num_frames}: focus at {focus_dist:.2f}m")
        
        from depth_utils import compute_coc
        coc = compute_coc(depth_map, focal_length, f_number, focus_dist,
                         sensor_width, W, max_blur)
        
        frame = render_dof(image, coc)
        
        frame_vis = (frame * 255).astype(np.uint8)
        cv2.putText(frame_vis, f"Focus: {focus_dist:.1f}m", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frames.append(frame_vis)
    
    return frames

def create_aperture_comparison(image, depth_map, focus_distance, 
                               focal_length, sensor_width, max_blur):
    from depth_utils import compute_coc
    
    apertures = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
    W = image.shape[1]
    H = image.shape[0]
    
    results = []
    for f_num in apertures:
        print(f"Rendering f/{f_num}...")
        
        from depth_utils import compute_coc
        coc = compute_coc(depth_map, focal_length, f_num, focus_distance,
                         sensor_width, W, max_blur)
        
        result = render_dof(image, coc)
        
        result_vis = (result * 255).astype(np.uint8)
        cv2.putText(result_vis, f"f/{f_num}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        results.append(result_vis)
    
    top_row = np.hstack(results[:3])
    bottom_row = np.hstack(results[3:])
    grid = np.vstack([top_row, bottom_row])
    
    return grid / 255.0

def save_image(image, path):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {path}")
def save_image(image, path):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}")

# import numpy as np
# import cv2

# def render_dof(image, coc_map, num_layers=20, max_radius=40):
#     """Basic depth-of-field rendering"""
#     H, W = image.shape[:2]
    
#     # Create blur layers
#     blur_radii = np.linspace(0, max_radius, num_layers)
#     blur_layers = []
    
#     print(f"Generating {num_layers} blur layers...")
#     for i, radius in enumerate(blur_radii):
#         if radius < 0.5:
#             blur_layers.append(image.copy())
#         else:
#             sigma = radius / 2.0
#             blurred = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
#             blur_layers.append(blurred)
#         print(f"  {i+1}/{num_layers}", end='\r')
#     print()
    
#     # Map CoC to layers
#     layer_indices = np.clip((coc_map / max_radius) * (num_layers - 1), 0, num_layers - 1)
#     idx_low = np.floor(layer_indices).astype(np.int32)
#     idx_high = np.ceil(layer_indices).astype(np.int32)
#     alpha = (layer_indices - idx_low).astype(np.float32)
    
#     # Blend layers
#     print("Blending...")
#     result = np.zeros_like(image)
#     for i in range(num_layers):
#         mask_low = (idx_low == i).astype(np.float32)
#         mask_high = (idx_high == i).astype(np.float32)
#         weight = mask_low * (1 - alpha) + mask_high * alpha
#         if weight.max() > 0:
#             result += blur_layers[i] * weight[:, :, np.newaxis]
    
#     return np.clip(result, 0.0, 1.0)

# def create_side_by_side(original, result, depth_map=None):
#     """Create before/after comparison"""
#     H, W = original.shape[:2]
    
#     if depth_map is not None:
#         # 3-way comparison
#         depth_vis = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()))
#         depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
#         depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB) / 255.0
        
#         combined = np.hstack([original, result, depth_colored])
        
#         # Add labels
#         combined = (combined * 255).astype(np.uint8)
#         cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.putText(combined, "With DOF", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.putText(combined, "Depth Map", (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         return combined / 255.0
#     else:
#         # 2-way comparison
#         combined = np.hstack([original, result])
#         combined = (combined * 255).astype(np.uint8)
#         cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.putText(combined, "With DOF", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         return combined / 255.0

# def create_focus_sweep(image, depth_map, focal_length, f_number, 
#                        sensor_width, max_blur, num_frames=10):
#     """Create focus sweep animation (near to far)"""
#     W = image.shape[1]
#     depths = np.linspace(depth_map.min(), depth_map.max(), num_frames)
    
#     frames = []
#     for i, focus_dist in enumerate(depths):
#         print(f"Frame {i+1}/{num_frames}: focus at {focus_dist:.2f}m")
        
#         from Project.Refocus.depth_utils_copy import compute_coc
#         coc = compute_coc(depth_map, focal_length, f_number, focus_dist,
#                          sensor_width, W, max_blur)
        
#         frame = render_dof(image, coc)
        
#         # Add focus indicator
#         frame_vis = (frame * 255).astype(np.uint8)
#         cv2.putText(frame_vis, f"Focus: {focus_dist:.1f}m", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         frames.append(frame_vis)
    
#     return frames

# def create_aperture_comparison(image, depth_map, focus_distance, 
#                                focal_length, sensor_width, max_blur):
#     """Compare different apertures"""
#     apertures = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
#     W = image.shape[1]
#     H = image.shape[0]
    
#     results = []
#     for f_num in apertures:
#         print(f"Rendering f/{f_num}...")
        
#         from Project.Refocus.depth_utils_copy import compute_coc
#         coc = compute_coc(depth_map, focal_length, f_num, focus_distance,
#                          sensor_width, W, max_blur)
        
#         result = render_dof(image, coc)
        
#         # Add label
#         result_vis = (result * 255).astype(np.uint8)
#         cv2.putText(result_vis, f"f/{f_num}", (10, 40),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
#         results.append(result_vis)
    
#     # Create grid (2x3)
#     top_row = np.hstack(results[:3])
#     bottom_row = np.hstack(results[3:])
#     grid = np.vstack([top_row, bottom_row])
    
#     return grid / 255.0

# def save_image(image, path):
#     """Save image (handles float or uint8)"""
#     if image.dtype == np.float32 or image.dtype == np.float64:
#         image = (image * 255).astype(np.uint8)
#     cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     print(f"✓ Saved: {path}")
