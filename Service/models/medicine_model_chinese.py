from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import json
from Service.config import drug_info_model, drug_model_url


class MedicineModelChinese():
    def __init__(self):
        self.model = self.model_initialize()
        self.prompt = self.prompt_initialize()
        self.medicine_info = self.load_medicine_context()
    def model_initialize(self):
        model = OllamaLLM(model = drug_info_model, base_url = drug_model_url)
        return model
    def load_medicine_context(self):
        with open(r'D:\backend_algorithm_blind_person_guidance\Service\drug_knowledge.json', 'r', encoding='utf-8') as f:
            medicines = json.load(f)

        medicines = medicines['main']
        return medicines
    def inquire_medicine(self,medicine_name):
        result = None
        for medicine in self.medicine_info:
            if medicine["药品名称"] == medicine_name:
                result = medicine
                break
            else: 
                result = "未找到该药物名称的信息。"
                break
        return result

        
    def prompt_initialize(self):
        chinese_template = """ 
        你是一副智能眼镜，识别出了面前的药品，现在需要将这款药的相关信息清晰、准确地传达给一位盲人。以下是药品的信息：

        药品名称：{medicine}
        药品详情：{details}

        请将这些信息整理成自然流畅的一段话，完整覆盖药品的主要成分、作用、用途、使用方法、注意事项等。不要加标题，不要有任何开场白或客套话，只需像一个语音助手一样，直接陈述内容，语言要准确、简洁、通顺，尽量简短一点最好控制在一小句内容。
        """

        prompt = ChatPromptTemplate.from_template(chinese_template)
        return prompt
    def invoke(self, medicine_name):
        context = self.inquire_medicine(medicine_name)

        
        chain = self.prompt | self.model
        response = chain.invoke({
        "medicine": medicine_name,
        "details": context})
        return response
