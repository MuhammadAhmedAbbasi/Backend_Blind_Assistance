import pyttsx3
import os

def text_to_speech_and_show_bytes(text, filename="output.mp3"):
    """Convert text to speech, save it, then read and display the bytes"""
    try:
        # Initialize the engine
        engine = pyttsx3.init()
        
        # Try to set Chinese voice if available
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'Chinese' in voice.name or 'ZH' in voice.languages:
                engine.setProperty('voice', voice.id)
                break
        
        # Save to file
        print(f"Saving speech to {filename}...")
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        # Read the file back as bytes
        print("\nReading the audio file bytes...")
        with open(filename, 'rb') as f:
            audio_bytes = f.read()
            
        # Display first 100 bytes (header) and last 100 bytes
        print("\nFirst 100 bytes:")
        print(audio_bytes[:100])
        
        print("\nLast 100 bytes:")
        print(audio_bytes[-100:])
        
        print(f"\nTotal file size: {len(audio_bytes)} bytes")
        
        # Return the bytes if you want to use them elsewhere
        return audio_bytes
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up - remove the file if you want
        if os.path.exists(filename):
            os.remove(filename)

# Example usage
if __name__ == "__main__":
    chinese_text = "你好，这是一个pyttsx3中文语音合成示例。"
    audio_bytes = text_to_speech_and_show_bytes(chinese_text)