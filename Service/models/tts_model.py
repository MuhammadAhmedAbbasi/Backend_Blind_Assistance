import pyttsx3
import os
import base64
from pydub import AudioSegment

from Service.base_models.base_tts import BaseTTS

class TTS(BaseTTS):
    def __init__(self):
        self.engine = pyttsx3.init()

    def text_to_speech_and_show_bytes(self, text: str, filename: str = "output.wav", mode: str = 'ZH', resampled_filename: str = "output_resampled.wav"):
        """Convert text to speech, resample, return bytes and base64"""
        try:
            if mode == 'ZH':
                input_voice, input_language = 'Chinese', 'ZH'
            else:
                input_voice, input_language = 'English', 'EN'

            # Initialize engine and select voice
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if input_voice in voice.name or input_language in voice.languages:
                    self.engine.setProperty('voice', voice.id)
                    break

            # Save original TTS audio
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()

            # Resample audio to 16 kHz
            sound = AudioSegment.from_wav(filename)
            sound = sound.set_frame_rate(16000)
            sound.export(resampled_filename, format="wav")

            # Read resampled file bytes
            with open(resampled_filename, 'rb') as f:
                audio_bytes = f.read()
            audio_bytes_64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_bytes_64, audio_bytes

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up original and resampled files if needed
            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(resampled_filename):
                os.remove(resampled_filename)

