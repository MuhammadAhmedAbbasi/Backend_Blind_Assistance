from pydantic import BaseModel
from typing import Optional, List

class ImageProcessingReturn(BaseModel):
    audio_bytes: Optional[str] = None
    answer: Optional[str] = None
    imp_image_info: Optional[list] = None
    other_image_info: Optional[list] = None
    resized_image: Optional[str] = None 
    medicine_info: Optional[str] = None
    mode_selection: Optional[str] = None
    ocr_detection: List[str] = None
    medicine_name: Optional[str] = None