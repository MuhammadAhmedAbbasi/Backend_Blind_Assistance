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
        tem = """   您作为专家，需要分析经过 OCR 从图片中识别出的名称列表。如果其中有一个名称是正确的药品名称，或者没有任何一个名称与原药品名称正确匹配或相关，那么您可以自行分析并给出正确的名称，具体如下：

                    OCR 给出的结果为以下列表：{input}

                    您需要从中选择的原药品名称如下：

                    1. 复方丹参滴丸
                    2. 复方氨酚烷胺片
                    3. 感冒灵胶囊
                    4. 正天丸
                    5.连花清瘟胶囊

                    如果该名称与上述名称完全不匹配，请给出以下名称：
                    6. 无匹配项
                    """

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
            detection.append("未检测到任何东西")
        return detection
    def invoke(self, image_bytes):
        detection = self.OCR_detection(image_bytes)
        res = self.prompt | self.correction_model
        answer = res.invoke({'input': detection}).medicine_name
        return answer
        
