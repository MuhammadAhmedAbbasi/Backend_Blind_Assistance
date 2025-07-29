from Service.model_service.detection_new import BlindDetection
import json

async def blind_glasses_handler(websocket):
    blind_guidance_model = BlindDetection()

    async for data in websocket:
        try:
            obstacles, _, _ = await blind_guidance_model.image_processing(data)
            response = {
                "audio": obstacles.audio_bytes,
                "text_command": obstacles.answer,
                "imp_image_info": obstacles.imp_image_info,
                "other_image_info": obstacles.other_image_info,
                "resized_image": obstacles.resized_image
            }
            await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))
