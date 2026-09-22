import os
import cv2
import sys
import asyncio

# Adjust paths to match your directory structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Service.common.image_processing_return import ImageProcessingReturn
from Service.model_service.detection_new import BlindDetection

# === CONFIGURATION ===
VIDEO_PATH = r"D:\backend_algorithm_blind_person_guidance\assets\test2.mp4"
SAVE_DIR = r'D:\backend_algorithm_blind_person_guidance\file_save_paths'
OUTPUT_VIDEO_NAME = "final_output_detection.mp4" 

IMAGE_EXTENSION = ".jpg"
AUDIO_EXTENSION = ".wav"
FILE_SAVE_INTERVAL_SECONDS = 1  # Save individual jpg/wav every 10 seconds

# === SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
blind_detection = BlindDetection()
mode = "detection"

def save_audio_bytes(audio_bytes: bytes, output_path: str):
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

async def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # 1. Get Video Properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 30 # Fallback
    
    # Calculate how often to save individual files (e.g., every 300 frames for 10s)
    frames_between_saves = int(original_fps * FILE_SAVE_INTERVAL_SECONDS)
    
    print(f"Video FPS: {original_fps}")
    print(f"Processing EVERY frame for video.")
    print(f"Saving individual JPG/WAV files every {frames_between_saves} frames ({FILE_SAVE_INTERVAL_SECONDS}s).")

    # Video Writer variables
    video_writer = None
    output_video_path = os.path.join(SAVE_DIR, OUTPUT_VIDEO_NAME)

    frame_count = 0
    saved_file_index = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------------------------------------------------
            # PROCESS EVERY FRAME (For the Video)
            # ---------------------------------------------------------
            
            # Encode frame to bytes
            success, encoded_image = cv2.imencode(IMAGE_EXTENSION, frame)
            if not success: continue
            image_bytes = encoded_image.tobytes()

            # Call service
            result, audio_bytes_raw, resized_frame = await blind_detection.image_processing(
                image_bytes=image_bytes, 
                glasses_mode=mode
            )

            # ---------------------------------------------------------
            # UPDATE VIDEO WRITER
            # ---------------------------------------------------------
            if resized_frame is not None:
                # Initialize writer on the first valid frame
                # We do this here because we need the dimensions of the *resized* frame
                if video_writer is None:
                    h, w, _ = resized_frame.shape
                    # 'mp4v' is standard. If this still fails, try 'avc1' or use .avi with 'MJPG'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, original_fps, (w, h))
                    print(f"Initialized Video Writer: {w}x{h} @ {original_fps} FPS")

                video_writer.write(resized_frame)

            # ---------------------------------------------------------
            # SAVE INDIVIDUAL FILES (Every 10 Seconds)
            # ---------------------------------------------------------
            if frame_count % frames_between_saves == 0:
                print(f"--> Saving individual checkpoint at frame {frame_count}...")
                
                # Save Image
                if resized_frame is not None:
                    image_filename = f"frame_{saved_file_index}{IMAGE_EXTENSION}"
                    image_path = os.path.join(SAVE_DIR, image_filename)
                    cv2.imwrite(image_path, resized_frame)
                    print(f"    Saved Image: {image_filename}")

                # Save Audio
                if audio_bytes_raw:
                    audio_filename = f"frame_{saved_file_index}{AUDIO_EXTENSION}"
                    audio_path = os.path.join(SAVE_DIR, audio_filename)
                    save_audio_bytes(audio_bytes_raw, audio_path)
                    print(f"    Saved Audio: {audio_filename}")
                
                saved_file_index += 1

            # Optional: Print progress every 30 frames so you know it's working
            if frame_count % 30 == 0:
                print(f"Processed frame {frame_count}...")

            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # ---------------------------------------------------------
        # CLEANUP (Crucial for video to play)
        # ---------------------------------------------------------
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"Video saved successfully to: {output_video_path}")
        else:
            print("Video writer was never initialized (no frames processed).")
            
        print("Processing finished.")

if __name__ == "__main__":
    asyncio.run(process_video(VIDEO_PATH))