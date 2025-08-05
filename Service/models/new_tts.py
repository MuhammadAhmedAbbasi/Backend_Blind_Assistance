import io
import base64
import soundfile as sf  # Use this to save NumPy audio as bytes (e.g., in WAV format)
from transformers import BarkModel, AutoProcessor
import numpy as np

class TextToAudio():
    def __init__(self):
        self.load_all_models()
        self.voice_preset = r"D:\Backend_insurance\Algorithm\audio\zh_speaker_7.npz"
    
    def load_all_models(self):
        model_path = r'D:\tts\bark-small'
        self.model = BarkModel.from_pretrained(model_path).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_path, padding=True, truncation=True)

    def generate(self, text):
        chunks = self.split_text(text)
        audio_segments = []

        for chunk in chunks:
            inputs = self.processor(chunk, voice_preset=self.voice_preset).to("cuda")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            output = self.model.generate(**inputs)
            audio_array = output.cpu().numpy().squeeze()
            audio_segments.append(audio_array)
        
        final_audio = np.concatenate(audio_segments)
        return final_audio

    def text_to_speech_and_show_bytes(self, audio_array, sample_rate=22050):
        # Save NumPy audio to in-memory WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()

        # Base64 encode
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64, audio_bytes

    def split_text(self, text, max_len=30):
        if len(text) <= max_len:
            return [text]
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + max_len
            window_end = min(end, text_length)

            last_full_stop = text.rfind('。', start, window_end)
            if last_full_stop != -1:
                end = last_full_stop + 1
            else:
                last_comma = text.rfind('，', start, window_end)
                if last_comma != -1:
                    end = last_comma + 1
                else:
                    end = window_end
            
            end = min(end, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks
