import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load model and image
yolo_path = r'D:\backend_algorithm_blind_person_guidance\yolo11x.pt'
image_path = r'D:\backend_algorithm_blind_person_guidance\other_materials\gaojinwei1.jpg'

image = cv2.imread(image_path)
w, h = 1500, 1000
angle = 0.707
dsize = (w, h)
image = cv2.resize(image, dsize=dsize)

height, width = image.shape[:2]
per_d = 40
nper_d = per_d/100

model = YOLO(yolo_path)

# Define allowed classes
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
    63: 'laptop'
}

# Calculate vicinity points
per_height = height - int(height*nper_d)
mid_width = width // 2 

length = per_height - height
angle = math.radians(45)
angle_width = int(mid_width + length * math.cos(angle))
angle_height = int(per_height - length * math.sin(angle))
#angle_height = int(height*angle)

#angle_width = int(width*angle)
print(f'Type of width: {angle_width}, and type of angle height: {angle_height}')
diff_width = (angle_width - mid_width)
op_angle_width = mid_width - diff_width
op_angle_width = int(mid_width - length * math.cos(angle))

# Create vicinity polygon
vicinity_points = [
    (mid_width, per_height),    # Top center
    (angle_width, angle_height),  # Right angle point
    (angle_width, height),      # Bottom right
    (mid_width, height),        # Bottom center
    (op_angle_width, height),   # Bottom left
    (op_angle_width, angle_height),  # Left angle point
]
vicinity_polygon = np.array(vicinity_points, np.int32)

def is_box_intersecting_polygon(box, polygon):
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
            if line_intersection(box_edge, poly_edge):
                return True

    return False

def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
def line_intersection(line1, line2):
    """Check if two line segments intersect"""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    
    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)
    D = (x4, y4)
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Process detections
result = model(image)[0]
objects_in_vicinity = []

for count, box in enumerate(result.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    
    # Skip if class not in allowed list or confidence < 0.7
    if class_id not in ALLOWED_CLASSES or confidence < 0.7:
        continue
    
    label = ALLOWED_CLASSES[class_id]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Check if object bounding box intersects with vicinity polygon
    is_inside = is_box_intersecting_polygon((x1, y1, x2, y2), vicinity_polygon)
    
    if is_inside:
        # Determine if object is on left or right side based on center
        position = "left" if center_x < mid_width else "right"
        objects_in_vicinity.append({
            'label': label,
            'position': position,
            'center': (center_x, center_y),
            'confidence': confidence,
            'box': (x1, y1, x2, y2)
        })
    
    # Draw box and info (regardless of vicinity)
    color = (0, 255, 0)  # Default green
    thickness = 2
    if is_inside:
        color = (0, 0, 255)  # Red for objects in vicinity
        thickness = 3
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    label_text = f"{label} {confidence:.2f}"
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    cv2.circle(image, (center_x, center_y), 3, color, -1)

# Draw vicinity area
pts = vicinity_polygon.reshape((-1, 1, 2))
cv2.polylines(image, [vicinity_polygon], isClosed=True, color=(0, 255, 255), thickness=3)

# Prepare labels for reference points
label_text = f'({mid_width}, {per_height})'
person_label = f'({mid_width}, {height})'
angle_label = f'({angle_width}, {angle_height})'
op_angle_label = f'({op_angle_width}, {angle_height})'
b_angle_label = f'({angle_width}, {height})'
b_op_angle_label = f'({op_angle_width}, {height})'

# Draw all reference points and labels as requested
cv2.circle(image, (mid_width, per_height), 5, (0, 0, 255), -1)
cv2.putText(image, label_text, (mid_width, per_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

cv2.circle(image, (mid_width, height), 5, (0, 0, 255), -1)
cv2.putText(image, person_label, (mid_width, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

cv2.circle(image, (angle_width, angle_height), 5, (0, 0, 255), -1)
cv2.putText(image, angle_label, (angle_width, angle_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

cv2.circle(image, (op_angle_width, angle_height), 5, (0, 0, 255), -1)
cv2.putText(image, op_angle_label, (op_angle_width, angle_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

cv2.circle(image, (angle_width, height), 5, (0, 0, 255), -1)
cv2.putText(image, b_angle_label, (angle_width, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

cv2.circle(image, (op_angle_width, height), 5, (0, 0, 255), -1)
cv2.putText(image, b_op_angle_label, (op_angle_width, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

# Draw all straight lines as requested
cv2.line(image, (mid_width, per_height), (mid_width, height), (0, 0, 255), 2)
cv2.line(image, (angle_width, angle_height), (mid_width, height), (0, 0, 255), 2)
cv2.line(image, (op_angle_width, angle_height), (mid_width, height), (0, 0, 255), 2)
cv2.line(image, (angle_width, height), (mid_width, height), (0, 0, 255), 2)
cv2.line(image, (op_angle_width, height), (mid_width, height), (0, 0, 255), 2)

# Display objects in vicinity information
info_y = 30
for obj in objects_in_vicinity:
    info_text = f"{obj['label']} detected on your {obj['position']} (confidence: {obj['confidence']:.2f})"
    cv2.putText(image, info_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    info_y += 30

if not objects_in_vicinity:
    cv2.putText(image, "No objects detected in your vicinity", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow('Blind Person Guidance System', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print results to console
print("\nObjects in vicinity analysis:")
if objects_in_vicinity:
    for obj in objects_in_vicinity:
        print(f"- {obj['label']} detected on your {obj['position']} (confidence: {obj['confidence']:.2f})")
else:
    print("- No objects detected in your vicinity")