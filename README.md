# Blind Assistance Backend

Backend services for an assistive vision system. The service accepts camera frames, detects nearby obstacles, generates navigation guidance, performs medicine recognition, and returns text, structured metadata, and optional audio.

## Capabilities

- YOLO object detection and Depth Anything V2 depth estimation.
- Chinese and English navigation guidance through Ollama.
- Medicine OCR and medicine-information lookup.
- Text-to-speech audio responses.
- REST and WebSocket interfaces.
- CPU fallback when CUDA is unavailable.
- Bounded inference concurrency and latest-frame processing for live clients.

## Requirements

- Python 3.10 or later.
- NVIDIA GPU with a compatible CUDA/PyTorch installation is recommended.
- Ollama for guidance and medicine-language models.
- Depth Anything V2 and YOLO model files.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA systems, install the PyTorch build matching the installed CUDA runtime by using the official PyTorch selector.

## Model Setup

Place the model files in the directories expected by `Service/models/detection_model.py`. The default configuration expects:

```text
Models/
├── Depth-Anything-V2/
│   └── depth_anything_v2/
└── Depth-Anything-V2-Small/
    └── depth_anything_v2_vits.pth
```

The default YOLO weight is `yolo11x.pt`. Model names and paths are controlled by `Service/config.json`.

Install and start Ollama:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

Update the Ollama URLs in `Service/config.json` when Ollama runs on another host.

## Configuration

Edit [Service/config.json](Service/config.json) for model names, Ollama URLs, image dimensions, confidence thresholds, language mode, and output locations. Generated files are kept under:

```text
runtime_data/
├── audio/
├── files/
└── images/
```

The directories are created automatically. Debug image output is disabled by default.

## Running

From the repository root:

```powershell
python Service\main.py
```

or:

```powershell
python -m Service.main
```

The service exposes HTTP on `http://localhost:8888` and WebSocket on `ws://localhost:8766`.

## REST API

### Blind guidance and medicine mode

`POST /algorithm/api/blind/detect/`

Multipart fields:

| Field | Required | Values |
|---|---:|---|
| `file` | Yes | Image upload |
| `glasses_mode` | Yes | `detection` or `Drug_detection` |

Example:

```powershell
curl.exe -X POST "http://localhost:8888/algorithm/api/blind/detect/" `
  -F "file=@sample.jpg" `
  -F "glasses_mode=detection"
```

### Dedicated medicine endpoint

`POST /algorithm/api/drug_detection/detect`

```powershell
curl.exe -X POST "http://localhost:8888/algorithm/api/drug_detection/detect" `
  -F "file=@medicine.jpg"
```

Responses contain base64 audio, text guidance, detection metadata, and optional medicine information.

## WebSocket API

Connect to `ws://localhost:8766` and send:

```json
{
  "image": "<base64 image>",
  "mode": "detection",
  "send_time": 1710000000000
}
```

Supported modes are `detection` and `Drug_detection`. The server sends periodic heartbeats. Live clients should send the newest frame; stale frames are dropped intentionally. The WebSocket message limit is 10 MB.

## Streamlit Test Client

The manual client is `Service/testing/testing.py`.

```powershell
pip install streamlit
streamlit run Service/testing/testing.py
```

It supports endpoint selection, timeouts, mode selection, JSON inspection, audio playback, and WAV download. Set `BLIND_API_URL` to provide a default endpoint.

## Docker

```powershell
docker build -t blind-assistance-backend .
docker run --gpus all -p 8888:8888 -p 8766:8766 blind-assistance-backend
```

For production GPU deployments, use an NVIDIA CUDA base image and provide model files and Ollama connectivity explicitly.

## Validation

```powershell
python -m compileall -q Service
```

Add mocked model tests, API contract tests, WebSocket disconnect tests, and GPU/CPU load tests before production deployment.

## Operational Notes

- Inference is serialized through a shared semaphore to protect model state and GPU memory.
- Blocking inference runs outside the async event loop.
- Invalid images return client errors; internal details remain in server logs.
- Add authentication, TLS, rate limiting, metrics, and health probes before exposing the service publicly.
- Validate depth calibration and navigation behavior with real-world tests before safety-critical use.

## Project Layout

```text
Service/
├── api/                   REST routes
├── base_models/           Abstract model interfaces
├── common/                Shared helpers
├── handler/               WebSocket handling
├── model_service/         Inference orchestration
├── models/                Detection, OCR, guidance, and TTS
├── testing/               Streamlit test client
├── config.json            Runtime configuration
└── main.py                Application entry point
```

Review the licenses for YOLO, Depth Anything V2, EasyOCR, Ollama models, and downloaded checkpoints before redistribution or commercial deployment.

