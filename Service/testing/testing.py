"""Streamlit client for manually testing the blind-assistance HTTP API."""

import base64
import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Blind Assistance Test Client", layout="wide")


def process_image(image_file: Any, api_url: str, mode: str, timeout: float) -> None:
    image_file.seek(0)
    files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
    try:
        with st.spinner("Processing image..."):
            response = requests.post(api_url, files=files, data={"glasses_mode": mode}, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        st.error(f"The API timed out after {timeout:g} seconds.")
        return
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to {api_url}")
        return
    except requests.exceptions.HTTPError:
        st.error(f"API returned HTTP {response.status_code}: {response.text[:1000]}")
        return
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return

    try:
        result = response.json()
    except ValueError:
        st.error("The API returned invalid JSON.")
        st.code(response.text[:2000])
        return

    st.success("API request successful")
    with st.expander("API response"):
        st.json(result)

    audio = result.get("audio")
    if not audio:
        st.info("No audio was returned.")
        return
    try:
        audio_bytes = base64.b64decode(audio, validate=True)
    except (ValueError, TypeError):
        st.warning("The API returned invalid base64 audio.")
        return
    st.audio(audio_bytes, format="audio/wav")
    st.download_button("Download audio", data=audio_bytes, file_name="blind_assistance_response.wav", mime="audio/wav")


st.title("Blind Assistance Test Client")
st.caption("Upload an image and exercise the deployed blind-assistance API.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("Detection API URL", value=os.getenv("BLIND_API_URL", "http://localhost:8888/algorithm/api/blind/detect/"))
    timeout = st.number_input("Request timeout (seconds)", min_value=5.0, max_value=300.0, value=60.0, step=5.0)
    mode = st.selectbox("Mode", ["detection", "Drug_detection"], format_func=lambda value: "Blind guidance" if value == "detection" else "Drug detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
    if st.button("Process image", type="primary", use_container_width=True):
        if not api_url.strip():
            st.error("Enter an API URL first.")
        else:
            process_image(uploaded_file, api_url.strip(), mode, timeout)

with st.expander("Usage"):
    st.markdown("1. Start the Service API.\n2. Confirm the URL and mode in the sidebar.\n3. Upload an image and select **Process image**.\n4. Inspect the JSON response or download the generated WAV audio.")

st.divider()
st.caption("Blind Assistance System - API test client")
