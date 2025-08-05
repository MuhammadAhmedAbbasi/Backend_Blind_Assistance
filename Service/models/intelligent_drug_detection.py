import easyocr
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
import cv2

class Structured_Output(BaseModel):
    medicine_name: str

class IntelligentDrugDetection:
    def __init__(self):
        self.reader, self.correction_model = self.model_initialize()
        self.correction_model = self.correction_model.with_structured_output(Structured_Output)
        self.prompt = self.correction_prompt()
    def model_initialize(self):
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        correction_model = ChatOllama(model = 'qwen2.5:7b', base_url = "http://localhost:11434")
        return reader, correction_model
    def correction_prompt(self):
        tem = """   You are the expert which will analyze the list of names detected from image after OCR, one of name is correct name of medicine or if not a single name is correct or relevant to the name of original medicine name, then you can analyze on your own and give proper name which:

                    The result given by OCR in form of is list following: {input}
                    The original names of medicine from which you have to select is below:
                    1. 复方丹参滴丸
                    2. 复方氨酚烷胺片
                    3. 感冒灵胶囊
                    4. 正天丸
                    5. 连花清瘟胶囊 """

        prompt = ChatPromptTemplate.from_template(tem)
        return prompt
    def OCR_detection(self, image):
        results = self.reader.readtext(image)
        detection = []
        if results:
            for bbox, text, conf in results:
                current_conf = float(conf)
                detection.append(f'{text} (Confidence: {current_conf:.2f})')
        else:
            print("Nothing Detected")
        return detection
    def invoke(self, image_bytes):
        detection = self.OCR_detection(image_bytes)
        res = self.prompt | self.correction_model
        answer = res.invoke({'input': detection}).medicine_name
        return answer
        
