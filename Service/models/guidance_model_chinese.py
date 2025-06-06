from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from Service.base_models.base_guidance_model import BaseGuidanceModel
from Service.config import llm_model, base_url


class GuidanceModelChinese(BaseGuidanceModel):
    def __init__(self):
        self.model = OllamaLLM(model=llm_model, base_url=base_url)
        self.chinese_template = self.prompt()
    def generate_scene_description(self, inside_vicinity, outside_vicinity):
        imp_scene_parts = []
        """Convert obstacle data into natural language description"""
        if not inside_vicinity:  # If list is empty
            imp_scene_description = "附近什么都没有，你可以直走。"
        else: 
            for obj in sorted(inside_vicinity, key=lambda x: x['depth']):  # Sort by proximity
                imp_scene_parts.append(
                    f"A {obj['label']} {obj['zone']} ({obj['depth']:.1f} depth)")
            imp_scene_description = " 附近障碍物: " + "\n".join(imp_scene_parts)

        if not outside_vicinity:  # If list is empty
            extr_scene_description = "远处没有任何物体。"  # "No objects far away"
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
            
            extr_scene_description = "远处有: " + "，".join(count_parts)  # "Far away: "
        
        return imp_scene_description, extr_scene_description
    def prompt(self):
        chinese_template = """ 
                    你是一个盲人导航助手，提供简洁，面向行动的指导。 分析场景，并准确地响应基于附近物体的清晰的导航指令。 仅在相关时包含有关对象的简短上下文。 分析附近的物体，并根据它建议行动，对于远的物体只告诉场景。 
                    分析"附近的物体"信息非常重要，所以要优先考虑这些信息
                        重要事项:
                        -如果人在右侧，建议往左侧移动。
                        -如果人在左边，建议往右边移动
                        -根据深度值，您可以建议向右/向左/轻微向右/轻微向左移动
                        -不要给我深度值


                        首先告诉现场，然后告诉采取行动。 

                                            当前场景:
                                            {scene}

                            现在提供精确的一行直接导航指导与场景信息 :
                            例子回应：有三个人靠近你，一个很远。 在你的左边有一个人稍微向右移动。"""
        return chinese_template
    def invoke(self, vicinity_scene, outside_scene):
        self.imp_scene_description, self.extr_scene_description = self.generate_scene_description(vicinity_scene, outside_scene)
        self.scene_description = f"近处物体: {self.imp_scene_description}"
        print(*f'self.scene_description: {self.scene_description}, self.imp_scene_description : {self.imp_scene_description}, self.extr_scene_description : {self.extr_scene_description}')
        
        prompt = ChatPromptTemplate.from_template(self.chinese_template)
        chain = prompt | self.model
        response = chain.invoke({
        "scene": self.scene_description})
        return response
