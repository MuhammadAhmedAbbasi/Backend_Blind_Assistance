import cv2
import torch
import os
import sys
from ultralytics import YOLO
import numpy as np
import math
import time

from Service.config import *
from Service.base_models.base_detection_model import BaseDetectionModel
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Models/Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2


class DetectionLogic(BaseDetectionModel):
    def __init__(self):
        self.base_url = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../', 'Models')
        self.yolo_path = os.path.join(self.base_url, 'detection_models', detection_model)
        self.depth_path = os.path.join(self.base_url, 'Depth-Anything-V2-Small', depth_model)
        self.yolo, self.depth_model = self.initialize_models()

    def initialize_models(self):
        # Initialize YOLO model
        yolo_model = YOLO(model=self.yolo_path)
        
        # Initialize Depth model
        depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        depth_model.load_state_dict(torch.load(
            self.depth_path,
            map_location='cuda'
        ))
        depth_model.cuda()
        depth_model.eval()
        
        return yolo_model, depth_model

    def resizing_image(self, image, width=1500, height=1000, interpolation_method=cv2.INTER_CUBIC):
        dsize = (width, height)
        resized_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_method)
        return resized_image

    def depth_normalize(self, depth_np):
        depth_norm = cv2.normalize(depth_np, None, normalize_alpha, normalize_beta, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return depth_colormap
    @staticmethod
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    @staticmethod
    def line_intersection(line1, line2):
        """Check if two line segments intersect"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        return DetectionLogic.ccw(A, C, D) != DetectionLogic.ccw(B, C, D) and DetectionLogic.ccw(A, B, C) != DetectionLogic.ccw(A, B, D)

    def is_box_intersecting_polygon(self, box, polygon):
        """Check if any part of the bounding box intersects with the polygon"""
        x1, y1, x2, y2 = box

        # Ensure polygon is a 2D array of (x, y) points
        polygon = np.array(polygon).reshape(-1, 2)

        # Check if any corner is inside the polygon
        for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return True

        # Check if any edge of the box intersects with the polygon
        box_edges = [
            [(x1, y1), (x2, y1)],
            [(x2, y1), (x2, y2)],
            [(x2, y2), (x1, y2)],
            [(x1, y2), (x1, y1)]
        ]

        # Convert polygon to list of edges
        poly_edges = []
        num_points = len(polygon)
        for i in range(num_points):
            pt1 = tuple(polygon[i])
            pt2 = tuple(polygon[(i + 1) % num_points])
            poly_edges.append((pt1, pt2))

        # Check for intersection between box and polygon edges
        for box_edge in box_edges:
            for poly_edge in poly_edges:
                if DetectionLogic.line_intersection(box_edge, poly_edge):
                    return True

        return False

    def get_zone(self, center_x, mid_width, vicinity = False):
        """Classify object into zones using depth values."""
        direction = "left" if center_x < mid_width else "right"
        if vicinity == True:
            area = "Near"
            return f' {direction} {area}'
        area = "Far"
        return f'{direction} {area}'
    
    def define_vicinity(self, image ,percentage_vicinity: int, angle_value: int):
        height, width = image.shape[:2]
        vicinity_range = percentage_vicinity/100
        per_height = height - int(height*vicinity_range)
        mid_width = width // 2 
        length = per_height - height
        angle = math.radians(angle_value)
        angle_width = int(mid_width + length * math.cos(angle))
        angle_height = int(per_height - length * math.sin(angle))
        diff_width = (angle_width - mid_width)
        op_angle_width = mid_width - diff_width
        op_angle_width = int(mid_width - length * math.cos(angle))
        vicinity_points = [
            (mid_width, per_height),    # Top center
            (angle_width, angle_height),  # Right angle point
            (angle_width, height),      # Bottom right
            (mid_width, height),        # Bottom center
            (op_angle_width, height),   # Bottom left
            (op_angle_width, angle_height),  # Left angle point
        ]
        vicinity_polygon = np.array(vicinity_points, np.int32)
        return vicinity_polygon, height, width , mid_width, per_height, angle_width, angle_height,op_angle_width


    def process_frame(self, frame):
        inside_vicinity = []
        outside_vicinity = []
        # Resize frame
        resized_frame = self.resizing_image(frame,image_width, image_height)
        vicinity_polygon, img_height, img_width , person_width, per_height, angle_width, angle_height, op_angle_width = self.define_vicinity(resized_frame,vicinity_percentage,vicinity_angle)
        # Get YOLO detections and Depth Anything detection
        result = self.yolo(resized_frame)[0]
        depth_np = self.depth_model.infer_image(resized_frame)
        if torch.is_tensor(depth_np):
            depth_np = depth_np.cpu().numpy()
        
        img_height, img_width = resized_frame.shape[:2]
        depth_resized = cv2.resize(depth_np, (resized_frame.shape[1], resized_frame.shape[0]))
        
        # Define allowed classes (YOLO class IDs)
        ALLOWED_CLASSES = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            62: 'tv',
            63: 'laptop',
            9: 'traffic light',
            10: 'fire hydrant',
            15: 'cat',
            16: 'dog',
            58: 'potted plant',
            59: 'bed',
            4: 'airplane',
            6: 'train',
            7: 'truck'
        }
        
        # Process detections
        for count, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Skip if class not in allowed list or confidence < 0.7
            if class_id not in ALLOWED_CLASSES or confidence < yolo_model_confidence:
                continue
            
            label = ALLOWED_CLASSES[class_id]  # Get label from allowed classes

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Get depth at object center
            window = depth_resized[center_y-2:center_y+3, center_x-2:center_x+3]  # 5x5
            depth_value = float(np.median(window)) if window.size > 0 else -1
            depth_value = round(depth_value, 2)
            is_inside = self.is_box_intersecting_polygon((x1, y1, x2, y2), vicinity_polygon)
            
            if is_inside:
                zone = self.get_zone(center_x, person_width, True)
                inside_vicinity.append({
                    'id': count,
                    'label': label,
                    'depth': float(depth_value),
                    'zone': zone,
                    'x1y1x2y2':[x1,y1,x2,y2]
                })
            else:
                zone = self.get_zone(center_x, person_width, False)
                obstacle = {
                    'id': count,
                    'label': label,
                    'depth': float(depth_value),
                    'zone': zone,
                    'x1y1x2y2':[x1,y1,x2,y2]
                }
                outside_vicinity.append(obstacle)
            # Display objects in vicinity information
            info_y = 30
            # Draw box and info
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} {depth_value} {zone} {confidence:.2f}"
            cv2.putText(resized_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            cv2.circle(resized_frame, (center_x, center_y), 3, (0, 0, 255), -1)
         
         
        # Prepare labels for reference points
        slabel_text = f'({person_width}, {per_height})'
        sperson_label = f'({person_width}, {img_height})'
        sangle_label = f'({angle_width}, {angle_height})'
        sop_angle_label = f'({op_angle_width}, {angle_height})'
        sb_angle_label = f'({angle_width}, {img_height})'
        sb_op_angle_label = f'({op_angle_width}, {img_height})'

        # Draw all reference points and labels as requested
        cv2.circle(resized_frame, (person_width, per_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, slabel_text, (person_width, per_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.circle(resized_frame, (person_width, img_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, sperson_label, (person_width, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.circle(resized_frame, (angle_width, angle_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, sangle_label, (angle_width, angle_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.circle(resized_frame, (op_angle_width, angle_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, sop_angle_label, (op_angle_width, angle_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.circle(resized_frame, (angle_width, img_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, sb_angle_label, (angle_width, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        cv2.circle(resized_frame, (op_angle_width, img_height), 5, (0, 0, 255), -1)
        cv2.putText(resized_frame, sb_op_angle_label, (op_angle_width, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # Draw all straight lines as requested
        cv2.line(resized_frame, (person_width, per_height), (person_width, img_height), (0, 0, 255), 2)
        cv2.line(resized_frame, (angle_width, angle_height), (person_width, img_height), (0, 0, 255), 2)
        cv2.line(resized_frame, (op_angle_width, angle_height), (person_width, img_height), (0, 0, 255), 2)
        cv2.line(resized_frame, (angle_width, img_height), (person_width, img_height), (0, 0, 255), 2)
        cv2.line(resized_frame, (op_angle_width, img_height), (person_width, img_height), (0, 0, 255), 2)

        for obj in inside_vicinity:
            ssinfo_text = f"{obj['label']} detected on your {obj['zone']}"
            cv2.putText(resized_frame, ssinfo_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 30

        if not inside_vicinity:
            cv2.putText(resized_frame, "No objects detected in your vicinity", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
     
        # Create depth visualization
        depth_vis = self.depth_normalize(depth_resized)
        cv2.polylines(resized_frame, [vicinity_polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        
        return resized_frame, depth_vis, outside_vicinity, inside_vicinity
