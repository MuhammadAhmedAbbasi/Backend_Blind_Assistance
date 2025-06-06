from abc import ABC, abstractmethod

class BaseGuidanceModel(ABC):
    @abstractmethod
    def generate_scene_description(self, inside_vicinity, outside_vicinity):
        pass
    
    @abstractmethod
    def prompt(self):
        pass
    @abstractmethod
    def invoke(self, vicinity_scene, outside_scene):
        pass