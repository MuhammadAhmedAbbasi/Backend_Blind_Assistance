import json
from pathlib import Path
import os

def load_config():
    """Loads and returns the JSON config file."""
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(main_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# Extract variables (nested structure)
detection_model_params = config["Detection_Model_Parameters"]
guidance_model = config['Guidance_model']
model_service = config['model_service']
glasses_mode_detection = config['glasses_mode']

# Model Loading
detection_model = detection_model_params["model_loading"]["detection_model"]
depth_model = detection_model_params["model_loading"]["depth_model"]
llm_model = guidance_model['llm_model']

# Image Processing
image_height = detection_model_params["image_processing"]["image_height"]
image_width = detection_model_params["image_processing"]["image_width"]
normalize_alpha = detection_model_params["image_processing"]["normalize_alpha"]
normalize_beta = detection_model_params["image_processing"]["normalize_beta"]

# Image Vicinity
vicinity_percentage = detection_model_params["image_vicinity"]["vicinity_percentage"]
vicinity_angle = detection_model_params["image_vicinity"]["vicinity_angle"]

# Other Parameters
yolo_model_confidence = detection_model_params["other_parameters"]["yolo_model_confidence"]

# Glasses Mode Selection
blind_guidance_mode = glasses_mode_detection["blind_guidance_mode"]
drug_detection_mode = glasses_mode_detection["drug_detection_mode"]

# URL LOADING
base_url = guidance_model['base_url']
mode = model_service['mode']
file_suffix = model_service["file_suffix"]
file_directory = model_service['file_directory']
file_prefix = model_service['file_prefix']
file_suffix = model_service['file_suffix']

if __name__ == "__main__":
    # Print all loaded variables
    print("Detection Model:", detection_model)
    print("Depth Model:", depth_model)
    print("Image Height:", image_height)
    print("Image Width:", image_width)
    print("Normalize Alpha:", normalize_alpha)
    print("Normalize Beta:", normalize_beta)
    print("Vicinity Percentage:", vicinity_percentage)
    print("Vicinity Angle:", vicinity_angle)
    print("YOLO Confidence:", yolo_model_confidence)