import cv2
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

def depth_model(model_path, image_path):
    # Define model and load weights
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(
       model_path,
        map_location='cuda'
    ))
    model.cuda()
    model.eval()

    # Read input image
    raw_img = cv2.imread(image_path)

    # Inference
    depth = model.infer_image(raw_img)  # H x W raw depth map

    # Convert depth to numpy and normalize
    depth_np = depth.cpu().numpy() if torch.is_tensor(depth) else depth

    return depth_np


def depth_normalize_plot(depth_np):
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    # Apply colormap for visualization
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    # Resize for display (optional)
    display_scale = 0.3  # Adjust to control display size
    h, w = depth_colormap.shape[:2]
    resized_colormap = cv2.resize(depth_colormap, (int(w * display_scale), int(h * display_scale)))

    # Show the depth map
    cv2.imshow("Depth Map (Resized)", resized_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



model_path = r'D:\backend_algorithm_blind_person_guidance\Depth-Anything-V2-Small\depth_anything_v2_vits.pth'
image_path = r'D:\backend_algorithm_blind_person_guidance\test.jpg'

depth_np = depth_model(model_path, image_path)
depth_normalize_plot(depth_np)
print(f' The Depth NP: {depth_np}')
