from Service.model_service.detection_new import BlindDetection
import json
import base64
import websockets
import asyncio

HEARTBEAT_INTERVAL = 5

async def send_heartbeat(websocket):
    while True:
        try:
            heartbeat_message = {
            "audio": None, 
            'text_command': None, 
            'imp_image_info': None, 
            'other_image_info': None,
            'resized_image': None,
            'mode_selection': None,
            'medicine_info': None}
            message = json.dumps(heartbeat_message)
            await websocket.send(message)
            print("heartbeat sent")

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            break
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def blind_glasses_handler(websocket):
    blind_guidance_model = BlindDetection()
    #asyncio.create_task(send_heartbeat(websocket))

    async for data in websocket:
        try:
            message = json.loads(data)
            image_base64 = message.get("image")
            image_bytes = base64.b64decode(image_base64)
            mode = message.get("mode")
            
            obstacles = await blind_guidance_model.image_processing(image_bytes, glasses_mode = mode)
            response = {
            "audio": obstacles.audio_bytes, 
            'text_command': obstacles.answer, 
            'imp_image_info': obstacles.imp_image_info, 
            'other_image_info': obstacles.other_image_info,
            'resized_image': obstacles.resized_image,
            'mode_selection': obstacles.mode_selection,
            'medicine_info': obstacles.medicine_info}
            print(f'The response is done')
            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))
