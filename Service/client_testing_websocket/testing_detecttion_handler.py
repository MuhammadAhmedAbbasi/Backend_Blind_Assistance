import asyncio
import websockets
import json

async def send_image_to_websocket(image_path: str):
    uri = "ws://localhost:8000"  # Match the server's WebSocket port

    async with websockets.connect(uri) as websocket:
        while True:
            # Read the image as bytes
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            # Send raw image bytes
            await websocket.send(image_bytes)

            # Wait for and parse the response
            response = await websocket.recv()
            parsed = json.loads(response)

            # Output result
            if "error" in parsed:
                print(f"Server error: {parsed['error']}")
            else:
                print("Response received:")
                print(f"Text Command: {parsed['text_command']}")
                print(f"Audio (base64): {parsed['audio'][:30]}...")  # print snippet
                print(f"Important Info: {parsed['imp_image_info']}")
                print(f"Other Info: {parsed['other_image_info']}")
                print(f"Image (base64): {parsed['resized_image'][:30]}...")  # print snippet

# Example usage
if __name__ == "__main__":
    asyncio.run(send_image_to_websocket(r"D:\backend_algorithm_blind_person_guidance\examples\example_2.jpg"))
