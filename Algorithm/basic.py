import cv2
import torch
import os
import sys
from ultralytics import YOLO
import numpy as np
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

def reading_file(file_url):
    image = cv2.imread(file_url)
    return image

def video_capture_and_procesing(video_url):
    cam = cv2.VideoCapture(0)



def resizing_image(image, width=1200, height=1000, interpolation_method=cv2.INTER_CUBIC):
    dsize = (width, height)
    resized_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_method)
    return resized_image

def plotting_image(plot_image, depth_image):
    cv2.imshow('Final Image', plot_image)
    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(0)

def model(model_path, image):
    model = YOLO(model=model_path)
    result = model(image)[0]
    return result

def depth_model(model_path, raw_image):
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(
       model_path,
        map_location='cuda'
    ))
    model.cuda()
    model.eval()
    depth = model.infer_image(raw_image)
    depth_np = depth.cpu().numpy() if torch.is_tensor(depth) else depth
    return depth_np

def depth_normalize(depth_np):
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    return depth_colormap

def get_zone(center_x, person_x, depth_value, depth_threshold=5.2):
    """Classify object into zones using depth values.
    Args:
        depth_value: Median depth from depth map (0=close, 1=far)
        depth_threshold: Value to separate near/far (tune as needed)
    """
    print(depth_value,depth_threshold )
    direction = "left" if center_x < person_x else "right"
    proximity = "far" if depth_value > depth_threshold else "near"
    return f"{proximity}_{direction}"

def processing(model_result, depth_resized, resized_image):
    img_height, img_width = resized_image.shape[:2]
    
    # Assume person is at bottom center (you may want to detect person instead)
    person_x = img_width // 2
    person_y = img_height  # Bottom of the image

    total_distance = math.sqrt((img_height - person_y)**2 + (img_width - person_x)**2)
    
    # Image center point (midpoint of the entire image)
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    
    # Loop over detections
    for box in model_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model_result.names[class_id]
        
        # Calculate center of bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        print(f"The object: {label} center is {center_x} and person center is {person_x}")
        # Get depth at object center
        window = depth_resized[center_y-2:center_y+3, center_x-2:center_x+3]  # 5x5
        depth_value = float(np.median(window)) if window.size > 0 else -1
        
        # Calculate distance from person to object center in image coordinates
        distance_from_person = math.sqrt((center_x - person_x)**2 + (center_y - person_y)**2)
        
        # Calculate angle relative to image center
        rel_to_center_x = center_x - img_center_x
        rel_to_center_y = img_center_y - center_y  # y increases downward
        angle_from_center = math.degrees(math.atan2(rel_to_center_y, rel_to_center_x))
        percentage_distance = (distance_from_person/total_distance)*10
        # Determine zone (near_left, near_right, far_left, far_right)
        zone = get_zone(center_x, person_x, depth_value)
        
        print(f"Object: {label}, Depth: {depth_value:.2f}, Zone: {zone}, "
              f"Distance from person: {distance_from_person:.2f}px, "
              f"Angle from center: {angle_from_center:.1f}°",
              f'The percentage of distance is {percentage_distance}')
        
        # Draw box and info
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} {zone}"
        cv2.putText(resized_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.circle(resized_image, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Draw reference points
    cv2.circle(resized_image, (person_x, person_y), 5, (255, 0, 0), -1)  # Person (blue)
    cv2.circle(resized_image, (img_center_x, img_center_y), 5, (0, 255, 255), -1)  # Image center (yellow)
    
    # Draw dividing lines
    cv2.line(resized_image, (img_center_x, 0), (img_center_x, img_height), (255, 255, 255), 1)  # Vertical
    cv2.line(resized_image, (0, img_center_y), (img_width, img_center_y), (255, 255, 255), 1)  # Horizontal
    
    return resized_image

# Code implementation:
image_url = r'D:\backend_algorithm_blind_person_guidance\test4.jpg'
yolo_path = r'D:\backend_algorithm_blind_person_guidance\yolo11x.pt'
depth_model_path = r'D:\backend_algorithm_blind_person_guidance\Depth-Anything-V2-Small\depth_anything_v2_vits.pth'

image = reading_file(image_url)
resized_image = resizing_image(image)
model_output = model(yolo_path, resized_image)
depth_np = depth_model(depth_model_path, resized_image)
depth_final = cv2.resize(depth_np, (resized_image.shape[1], resized_image.shape[0]))
final_image = processing(model_output, depth_final, resized_image)
final_depth_image = depth_normalize(depth_final)

# Show final result
plotting_image(final_image, final_depth_image)




