import cv2
import torch
import os
import sys
from ultralytics import YOLO
import numpy as np
import math
import time
import pyttsx3

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

sys.path.append(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

def initialize_models(yolo_path, depth_model_path):
    # Initialize YOLO model
    yolo_model = YOLO(model=yolo_path)
    
    # Initialize Depth model
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(
        depth_model_path,
        map_location='cuda'
    ))
    depth_model.cuda()
    depth_model.eval()
    
    return yolo_model, depth_model

def resizing_image(image, width=1200, height=1000, interpolation_method=cv2.INTER_CUBIC):
    dsize = (width, height)
    resized_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_method)
    return resized_image

def depth_normalize(depth_np):
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    return depth_colormap

def get_zone(center_x, person_x, depth_value, depth_threshold=2.5):
    """Classify object into zones using depth values."""
    direction = "left" if center_x < person_x else "right"
    proximity = "near" if depth_value > depth_threshold else "far"
    return f"{proximity} {direction}"

def process_frame(yolo_model, depth_model, frame):
    # Resize frame
    resized_frame = resizing_image(frame)
    
    # Get YOLO detections
    result = yolo_model(resized_frame)[0]
    
    # Get depth map
    depth_np = depth_model.infer_image(resized_frame)
    if torch.is_tensor(depth_np):
        depth_np = depth_np.cpu().numpy()
    depth_resized = cv2.resize(depth_np, (resized_frame.shape[1], resized_frame.shape[0]))
    
    img_height, img_width = resized_frame.shape[:2]
    person_x = img_width // 2
    person_y = img_height  # Bottom of the image
    total_distance = math.sqrt((img_height - person_y)**2 + (img_width - person_x)**2)
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    frame_obstacles = []
    
    # Process detections
    for count , box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = result.names[class_id]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get depth at object center
        window = depth_resized[center_y-2:center_y+3, center_x-2:center_x+3]  # 5x5
        depth_value = float(np.median(window)) if window.size > 0 else -1
        depth_value = round(depth_value, 2)
        # Calculate distance and angle
        distance_from_person = math.sqrt((center_x - person_x)**2 + (center_y - person_y)**2)
        distance_from_person = round(distance_from_person, 2)  # Rounds to 2 decimal places
        rel_to_center_x = center_x - img_center_x
        rel_to_center_y = img_center_y - center_y
        angle_from_center = math.degrees(math.atan2(rel_to_center_y, rel_to_center_x))
        
        # Determine zone
        zone = get_zone(center_x, person_x, depth_value)

        obstacle = {
            'id': count,
            'object': label,
            'depth': float(depth_value),
            'zone': zone,
            'distance_px': float(distance_from_person),
            'angle_deg': float(angle_from_center),
            'timestamp': time.time()
        }
        frame_obstacles.append(obstacle)
        # Draw box and info
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} {depth_value} {zone}  {distance_from_person}"
        cv2.putText(resized_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.circle(resized_frame, (center_x, center_y), 3, (0, 0, 255), -1)
    # Draw reference points and lines
    cv2.circle(resized_frame, (person_x, person_y), 5, (255, 0, 0), -1)
    cv2.circle(resized_frame, (img_center_x, img_center_y), 5, (0, 255, 255), -1)
    cv2.line(resized_frame, (img_center_x, 0), (img_center_x, img_height), (255, 255, 255), 1)
    cv2.line(resized_frame, (0, img_center_y), (img_width, img_center_y), (255, 255, 255), 1)
    
    # Create depth visualization
    depth_vis = depth_normalize(depth_resized)
    
    return resized_frame, depth_vis, frame_obstacles

def llm(scene):
    template = """ 
    You are a blind navigation assistant. Analyze the scene and your answer should be in one line precise, 
    few words on objects near and then action. make it in professional guidance like answer:

        suggested commands which you can also use:
        1. "Move forward" - if path is clear
        2. "Turn left/right/slight right/slight left" - if obstacles are on one side
        
        Scene:
        {scene}
        
        Additional Rules:
        - Don't tell the angle values 
        - Don't tell exact values of distance, just mention if it is near or far and action to take
        - if people are right suggest slight left in order to avoid collosion similarly for other direction


        example answer:  There is a person on your right near, move slightly left

    """
    chinese_template = """你是一名盲人导航助手。请分析场景并用一行简洁精准的描述（说明附近物体及行动建议），以专业导航指令的形式回答：

                    建议指令（可选用）：

                    “直行”——若路径畅通

                    “左转, 右转, 微左转, 微右转”——若单侧有障碍物

                    场景：
                    {scene}

                    附加规则：

                    - 不提及具体角度

                    - 不提供精确距离，仅用“近”或“远”描述并建议行动

                    - 如右侧有人，建议“微左转”避让（其他方向同理）

                    示例回答： 右侧近处有行人，请微左转"""
    # LangChain setup
    prompt = ChatPromptTemplate.from_template(chinese_template)
    model = OllamaLLM(model="qwen2.5:3b", base_url="http://localhost:11434")
    chain = prompt | model

    # Run inference
    response = chain.invoke({
        "scene": scene
    })
    return response



def tts(text):
    # Initialize the engine
    engine = pyttsx3.init()

    # Set Chinese voice explicitly
    for voice in engine.getProperty('voices'):
        if 'huihui' in voice.name.lower():  # Find Microsoft Huihui
            engine.setProperty('voice', voice.id)
            break

    # Configure speech (optimized for Chinese)
    engine.setProperty('rate', 200)  # Slightly slower for clearer tones
    engine.setProperty('volume', 1.0)

    # Speak Chinese text
    engine.say(text)
    engine.runAndWait()

def tts_english(text):
    # Initialize the engine
    engine = pyttsx3.init()

    # Set English voice explicitly
    for voice in engine.getProperty('voices'):
        if 'english' in voice.name.lower() or 'en-us' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    # Configure speech (optimized for English)
    engine.setProperty('rate', 200)  # Moderate speaking rate
    engine.setProperty('volume', 1.0)  # Full volume
    engine.setProperty('pitch', 120)  # Slightly higher pitch for clarity

    # Speak English text
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(obstacles):
    """Convert obstacle data into natural language description"""
    scene_parts = []
    
    for obj in sorted(obstacles, key=lambda x: x['depth']):  # Sort by proximity
        scene_parts.append(
            f"A {obj['object']} {obj['zone']} ({obj['depth']:.1f} depth, "
            f"{obj['distance_px']:.0f} cm away, angle {obj['angle_deg']:.1f}°)"
        )
    
    return "Current scene contains: " + "\n".join(scene_parts)

def main():
    # Paths to models
    yolo_path = r'D:\backend_algorithm_blind_person_guidance\yolo11x.pt'
    depth_model_path = r'D:\backend_algorithm_blind_person_guidance\Depth-Anything-V2-Small\depth_anything_v2_vits.pth'
    
    # Initialize models
    yolo_model, depth_model = initialize_models(yolo_path, depth_model_path)
    
    # Initialize video capture (0 for webcam, or file path for video file)
    video_source = r'D:\backend_algorithm_blind_person_guidance\other_materials\WeChat_20250527184828.mp4'
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    llm_processing_interval = 40  # Process every 10th frame for LLM
    frame_counter = 0
    last_llm_time = 0
    llm_cooldown = 3  # Minimum seconds between LLM commands
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Process frame
            processed_frame, depth_frame, frame_obstacles = process_frame(yolo_model, depth_model, frame)

            # Only process for LLM at intervals AND if cooldown passed
            current_time = time.time()
            if (frame_counter % llm_processing_interval == 0 and 
                current_time - last_llm_time > llm_cooldown):
                scene = generate_scene_description(frame_obstacles)
                response_llm = llm(scene)
                tts(response_llm)
                print(f"LLM response : {response_llm}")
                last_llm_time = current_time
            frame_counter += 1
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow('Object Detection', processed_frame)
            cv2.imshow('Depth Map', depth_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()