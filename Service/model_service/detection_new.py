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
from Service.config import mode, file_suffix, file_prefix, file_directory, drug_detection_mode, blind_guidance_mode
from Service.models.intelligent_drug_detection import IntelligentDrugDetection
from Service.models.medicine_model_chinese import MedicineModelChinese
import logging
import time
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlindDetection:
    def __init__(self):
        self.detection_model = DetectionLogic()
        self.guidance_model_chinese = GuidanceModelChinese()
        self.guidance_model_english = GuidanceModelEnglish()
        self.intelli = IntelligentDrugDetection()
        self.perscription_model = MedicineModelChinese()
        self.tts_model = TTS()

    async def image_processing(self, image_path: str = None, image_bytes: bytes = None, glasses_mode: str = "detection") -> ImageProcessingReturn:
        save_dir_path = os.path.join((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), file_directory)
        if image_bytes != None:
            # Convert bytes to OpenCV format (numpy array)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_path != None:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            logging.error(f'No path or bytes found, kindly upload valid path or bytes')
            pass
        if glasses_mode == blind_guidance_mode:
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

            guidance_result  = model.invoke(inside_vicinity, outside_vicinity)
            guidance_audio_bytes_base64, guidanece_model_bytes = self.tts_model.text_to_speech_and_show_bytes(guidance_result)
            return ImageProcessingReturn(
                audio_bytes = guidance_audio_bytes_base64,
                answer = guidance_result,
                imp_image_info = inside_vicinity,
                other_image_info = outside_vicinity,
                resized_image = resized_image_base64,
                mode_selection = glasses_mode,
                medicine_info = None
            )
        

        elif glasses_mode == drug_detection_mode:
            start_time = time.time()
            medicine_name, ocr_detection =  self.intelli.invoke(img)
            end_ocr = time.time() - start_time
            logger.info(f' The Intelligent mode time is: {end_ocr}')
            if medicine_name == '无匹配项':
                medicine_info = '所给药物不在数据库中'
            elif medicine_name == '未检测到任何东西':
                medicine_info = '未检测到药物，请将药物放在摄像头前以获得清晰的视图'
            else:
                medicine_response_time = time.time()
                medicine_info = self.perscription_model.invoke(medicine_name)
                e_c_name = time.time() - medicine_response_time
            medicine_audio_bytes_base64, medicine_model_bytes = self.tts_model.text_to_speech_and_show_bytes(medicine_info)
            return ImageProcessingReturn(
                audio_bytes = medicine_audio_bytes_base64,
                mode_selection = glasses_mode,
                medicine_info = medicine_info,
                ocr_detection = ocr_detection,
                medicine_name =  medicine_name)
        

        else:
            return ImageProcessingReturn(
                mode_selection = "Invalid Mode Selected. Please choose 'detection' or 'Drug_detection'."
            )


    
blind_algo = BlindDetection()