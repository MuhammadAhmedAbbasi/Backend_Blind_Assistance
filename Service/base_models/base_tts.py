from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @abstractmethod
    def text_to_speech_and_show_bytes(self, text: str, filename: str="output.mp3", input_voice: str = 'Chinese', input_language: str = 'ZH'):
        pass