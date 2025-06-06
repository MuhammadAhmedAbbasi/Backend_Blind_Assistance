import cv2
import os 

from datetime import datetime


def save_image(image, base_path, prefix=""):
    """Helper function to save an image with a timestamp"""
    if image is not None:
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}{timestamp}.jpg"
        filepath = os.path.join(base_path, filename)
        
        # Save the image
        cv2.imwrite(filepath, image)
        print(f"Saved image to {filepath}")
        return filepath
    return None