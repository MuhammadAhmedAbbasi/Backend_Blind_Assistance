import os
import cv2
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Service.common.image_processing_return import ImageProcessingReturn
from Service.model_service.detection import image_processing

# === CONFIGURATION ===
VIDEO_PATH = r"D:\backend_algorithm_blind_person_guidance\testing_video_part\gaojinwei.mp4"  # Change this to your actual video path
FRAME_INTERVAL_SECONDS = 1  # Extract frame every 5 seconds
SAVE_DIR = r'D:\backend_algorithm_blind_person_guidance\photos_audio'
IMAGE_EXTENSION = ".jpg"
AUDIO_EXTENSION = ".wav"

# === MAKE SURE SAVE DIR EXISTS ===
os.makedirs(SAVE_DIR, exist_ok=True)

def save_audio_bytes(audio_bytes: bytes, output_path: str):
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    saved_frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame at {frame_count}...")

        # Encode frame to bytes (simulate camera input or HTTP request body)
        success, encoded_image = cv2.imencode(IMAGE_EXTENSION, frame)
        if not success:
            print("Error encoding frame.")
            continue

        image_bytes = encoded_image.tobytes()
        result, audio_bytes_raw, resized_frame = image_processing(image_bytes)

        # Save image
        image_filename = f"frame_{saved_frame_index}{IMAGE_EXTENSION}"
        image_path = os.path.join(SAVE_DIR, image_filename)
        with open(image_path, "wb") as img_file:
            img_file.write(resized_frame)
        
        print(f"darta: {result.imp_image_info}")
        
        # Only save audio if imp_image_info is not None or empty
        if result.imp_image_info:  # This checks for both None and empty list
            # Save audio
            audio_filename = f"frame_{saved_frame_index}{AUDIO_EXTENSION}"
            audio_path = os.path.join(SAVE_DIR, audio_filename)
            save_audio_bytes(audio_bytes_raw, audio_path)
            print(f"Saved audio: {audio_filename}")
        else:
            print("No image info detected, skipping audio save")

        print(f"Saved image: {image_filename}")
        saved_frame_index += 1
        frame_count += 1

    cap.release()
    print("Video processing completed.")
if __name__ == "__main__":
    process_video(VIDEO_PATH)
