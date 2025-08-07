import pyttsx3
import os
import base64
import tempfile
import gc
from pydub import AudioSegment
from base_models.base_tts import BaseTTS


class TTS(BaseTTS):
    def text_to_speech_and_show_bytes(self, text: str):
        """Convert Chinese text to speech, return bytes and base64"""
        filename = None
        resampled_filename = None

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                filename = tmp_wav.name
                print("Temp file created:", filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_resampled:
                resampled_filename = tmp_resampled.name
                print("Resampled file created:", resampled_filename)

            # Init new engine each time
            engine = pyttsx3.init()
            for voice in engine.getProperty('voices'):
                if 'Huihui' in voice.name or 'Chinese' in voice.name or 'ZH' in str(voice.languages):
                    engine.setProperty('voice', voice.id)
                    break

            engine.save_to_file(text, filename)
            engine.runAndWait()
            engine.stop()
            del engine
            gc.collect()
            print("Speech synthesis completed")

            # Resample to 16kHz
            sound = AudioSegment.from_wav(filename)
            print("Original sound loaded")
            sound = sound.set_frame_rate(16000)
            sound.export(resampled_filename, format="wav")
            print("Sound resampled")

            # Read bytes and base64
            with open(resampled_filename, 'rb') as f:
                print("Reading final audio file")
                audio_bytes = f.read()
                print(f"Audio length: {len(audio_bytes)}")
            audio_bytes_64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_bytes_64, audio_bytes

        except Exception as e:
            print(f"Error: {e}")
            return None, None

        finally:
            if filename and os.path.exists(filename):
                os.remove(filename)
            if resampled_filename and os.path.exists(resampled_filename):
                os.remove(resampled_filename)
