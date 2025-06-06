from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from Service.models.guidance_model_chinese import GuidanceModelChinese
from Service.config import llm_model, base_url
class GuidanceModelEnglish(GuidanceModelChinese):
    def __init__(self):
        self.model = OllamaLLM(model=llm_model, base_url=base_url)
        self.english_template = self.prompt()
    def generate_scene_description(self, inside_vicinity, outside_vicinity):
        imp_scene_parts = []
        """Convert obstacle data into natural language description"""
        if not inside_vicinity:  # If list is empty
            imp_scene_description = "There is nothing nearby, you can just walk straight ahead."
        else: 
            for obj in sorted(inside_vicinity, key=lambda x: x['depth']):  # Sort by proximity
                imp_scene_parts.append(
                    f"A {obj['label']} {obj['zone']} ({obj['depth']:.1f} depth)")
            imp_scene_description = "" + "\n".join(imp_scene_parts)

        if not outside_vicinity:  # If list is empty
            extr_scene_description = "There are no objects in the distance."  # "No objects far away"
        else: 
            # Count objects by label in outside_vicinity
            object_counts = {}
            for obj in outside_vicinity:
                label = obj['label']
                object_counts[label] = object_counts.get(label, 0) + 1
            
            # Create description parts
            count_parts = []
            for label, count in object_counts.items():
                if count == 1:
                    count_parts.append(f"1个{label}")  # "1 [label]"
                else:
                    count_parts.append(f"{count}个{label}")  # "[count] [label]s"
            
            extr_scene_description = "" + "，".join(count_parts)  # "Far away: "
        
        return imp_scene_description, extr_scene_description
    def prompt(self):
        english_template = """ 
            You are a blind navigation assistant that provides concise, action-oriented guidance. Analyze the scene and accurately respond to clear navigation instructions based on nearby objects. Include a short context about the object only when relevant. Analyze nearby objects and suggest actions based on them, and only tell the scene about distant objects.
                    It is very important to analyze the information of "nearby objects", so this information should be prioritized.
                        Important matters:
                        -If the person is on the right, it is recommended to move to the left.
                        -If the person is on the left, it is recommended to move to the right
                        -Depending on the depth value, you can suggest to move to the right/left/slightly to the right/slightly to the left
                        -Don't give me depth value


                        First tell the scene, then tell the action. 

                                            Current scene:
                                            {scene}

                        Now provide accurate one-line direct navigation guidance and scene information :
                        Example response: There are three people close to you, one is far away. There is a person on your left who moves slightly to the right.
                     """
        return english_template
    def invoke(self, vicinity_scene, outside_scene):
        self.imp_scene_description, self.extr_scene_description = self.generate_scene_description(vicinity_scene, outside_scene)
        
        # Combine the descriptions for the prompt
        self.scene_description = f"Nearby Objects: {self.imp_scene_description}"    
        prompt = ChatPromptTemplate.from_template(self.english_template)
        chain = prompt | self.model
        response = chain.invoke({
        "scene": self.scene_description})
        return response
