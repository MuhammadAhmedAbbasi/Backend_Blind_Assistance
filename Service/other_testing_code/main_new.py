import uvicorn
import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from Service.models.detection_model import DetectionLogic
from Service.config import image_height, image_width

if __name__ == "__main__":
    # Initialize detector
    detector = DetectionLogic()

    # Path to your input video
    input_video_path = r"D:\backend_algorithm_blind_person_guidance\testing_video_part\gao2.mp4"  # Replace with your video path
    output_video_path = r"D:\backend_algorithm_blind_person_guidance\testing_video_part\gaojinwei_detect_2.mp4"  # Output video path

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'mp4v' for .mp4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (image_width, image_height))  # Output resolution should match processed_frame

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left

        processed_frame, depth_vis, outside_vicinity, inside_vicinity = detector.process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Optional: Display while processing
        cv2.imshow("Processed Frame", processed_frame)
        cv2.imshow("Depth Map", depth_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing completed. Video saved to {output_video_path}")