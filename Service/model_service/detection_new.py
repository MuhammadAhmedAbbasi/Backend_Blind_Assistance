import cv2
import numpy as np
import os
from Service.models.detection_model import DetectionLogic
from Service.models.guidance_model_chinese import GuidanceModelChinese
from Service.models.guidance_model_english import GuidanceModelEnglish
from Service.models.new_tts import TextToAudio
from Service.models.tts_model import TTS
import base64
from Service.common.save_image import save_image
from Service.common.image_processing_return import ImageProcessingReturn
from Service.config import mode, file_suffix, file_prefix, file_directory


class BlindDetection:
    def __init__(self):
        self.detection_model = DetectionLogic()
        self.guidance_model_chinese = GuidanceModelChinese()
        self.guidance_model_english = GuidanceModelEnglish()
        self.tts_model = TTS()

    async def image_processing(self, image_bytes: bytes) -> ImageProcessingReturn:
        save_dir_path = os.path.join((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), file_directory)
        # Convert bytes to OpenCV format (numpy array)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized_frame, depth_vis, outside_vicinity, inside_vicinity = self.detection_model.process_frame(img)
        save_image(resized_frame, save_dir_path, file_prefix)
        # Convert resized_frame to bytes
        _, encoded_image = cv2.imencode(file_suffix, resized_frame)
        resized_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        guidance_models = {
        'ZH': self.guidance_model_chinese,
        'EN': self.guidance_model_english
    }

        model = guidance_models.get(mode)
        if not model:
            raise ValueError("Invalid mode selected. Choose 'ZN' or 'EN'.")

        guidance_result = model.invoke(inside_vicinity, outside_vicinity)
        guidance_audio_bytes_base64, guidanece_model_bytes = self.tts_model.text_to_speech_and_show_bytes(guidance_result, mode = mode)
        
        return ImageProcessingReturn(
            audio_bytes=guidance_audio_bytes_base64,
            answer=guidance_result,
            imp_image_info=inside_vicinity,
            other_image_info=outside_vicinity,
            resized_image=resized_image_base64
        ), guidanece_model_bytes, encoded_image
    
blind_algo = BlindDetection()