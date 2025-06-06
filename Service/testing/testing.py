import streamlit as st
import requests
import json
import base64
import soundfile as sf
import sounddevice as sd
import io
from pathlib import Path

# Streamlit UI setup
st.set_page_config(page_title="Image to Audio Converter", layout="wide")
st.title("Blind Assistance Image to Audio Converter")

def play_base64_audio(audio_base64):
    """Play base64 audio using sounddevice"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_file)
        sd.play(data, samplerate)
        sd.wait()
        return True
    except Exception as e:
        st.error(f"Error playing audio: {e}")
        return False

def process_image(image_file):
    """Send image to API and handle response"""
    # url = "http://localhost:8000/algorithm/api/blind/detect/"
    url = "http://192.168.1.100:8000/algorithm/api/blind/detect/"
    
    try:
        with st.spinner("Processing image..."):
            files = {"file": image_file}
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            response_data = response.json()
            st.success("API request successful!")
            
            # Display the JSON response
            with st.expander("API Response Data"):
                st.json(response_data)
            
            # Save the response
            json_filename = "response.json"
            with open(json_filename, "w") as f:
                json.dump(response_data, f, indent=4)
            
            # Check for audio data and play
            if 'audio' in response_data:
                st.audio(base64.b64decode(response_data['audio']), format='audio/wav')
                if st.button("Play Audio Response"):
                    play_base64_audio(response_data['audio'])
            else:
                st.warning("No audio data found in response")
            
            return response_data
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except json.JSONDecodeError:
        st.error(f"Invalid JSON response: {response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

# Main file uploader
uploaded_file = st.file_uploader(
    "Upload an image for audio description",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file to get audio description"
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process the image when user clicks the button
    if st.button("Generate Audio Description"):
        # Need to rewind the file pointer as Streamlit uploads are read-once
        uploaded_file.seek(0)
        process_image(uploaded_file)

# Instructions section
with st.expander("How to use this app"):
    st.markdown("""
    1. Upload an image file (JPG/PNG)
    2. Click 'Generate Audio Description' button
    3. The app will:
       - Send the image to the API
       - Display the JSON response
       - Show an audio player with the response
       - Provide a play button to hear the description
    4. Make sure your API server is running at `http://localhost:8000`
    """)

# Footer
st.markdown("---")
st.caption("Blind Assistance System - Image to Audio Converter")