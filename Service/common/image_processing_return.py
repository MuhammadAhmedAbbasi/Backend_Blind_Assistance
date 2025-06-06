from pydantic import BaseModel
from typing import Optional

class ImageProcessingReturn(BaseModel):
    audio_bytes: Optional[str] = None
    answer: Optional[str] = None
    imp_image_info: Optional[list] = None
    other_image_info: Optional[list] = None
    resized_image: Optional[str] = None 