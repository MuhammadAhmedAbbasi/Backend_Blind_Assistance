import pyttsx3
import os
import base64

from Service.base_models.base_tts import BaseTTS

class TTS(BaseTTS):
    def __init__(self):
        self.engine = pyttsx3.init()

    def text_to_speech_and_show_bytes(self, text: str, filename: str="output.mp3", mode: str = 'ZH'):
        """Convert text to speech, save it, then read and display the bytes"""
        try:
            if mode == 'ZH':
                input_voice, input_language = 'Chinese', 'ZH'
            else:
                input_voice, input_language = 'English', 'EN'

            
            # Initialize the engine
            self.engine = pyttsx3.init()
            
            # Try to set Chinese voice if available
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if input_voice in voice.name or input_language in voice.languages:
                    self.engine.setProperty('voice', voice.id)
                    break
            
            # Save to file
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            
            # Read the file back as bytes

            with open(filename, 'rb') as f:
                audio_bytes = f.read()
            audio_bytes_64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_bytes_64
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up - remove the file if you want
            if os.path.exists(filename):
                os.remove(filename)

