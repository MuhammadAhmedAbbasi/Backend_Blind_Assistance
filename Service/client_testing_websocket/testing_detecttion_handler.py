import asyncio
import websockets
import json
import base64

async def send_image_to_websocket(image_path: str):
    uri = "ws://localhost:8766"  # Match the server's WebSocket port

    async with websockets.connect(uri,max_size=2*1024*1024) as websocket:
        while True:
            # Read the image bytes
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            # Base64 encode the bytes to string
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Prepare JSON message
            message = {
                "image": image_base64,
                "mode": "Drug_detection"  # replace or remove as needed
            }

            # Send JSON as text
            await websocket.send(json.dumps(message))

            # Receive response
            response = await websocket.recv()
            print(f"Received response:")
if __name__ == "__main__":
    asyncio.run(send_image_to_websocket(r"D:\backend_algorithm_blind_person_guidance\Algorithm\training\new_data\images\medicine_1\Screenshot_20250729144720.png"))
