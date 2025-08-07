from Service.model_service.detection_new import BlindDetection
import json
import base64
import websockets
import asyncio
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 5
SAVE_DIR = r"D:\backend_algorithm_blind_person_guidance\json_save"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

async def send_heartbeat(websocket):
    while True:
        try:
            heartbeat_message = {
                "audio": None,
                'text_command': None,
                'imp_image_info': None,
                'other_image_info': None,
                'resized_image': None,
                'mode_selection': "Heartbeat",
                'medicine_info': None
            }
            message = json.dumps(heartbeat_message)
            await websocket.send(message)
            logger.info("Heartbeat sent")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during heartbeat")
            break
        except Exception as e:
            logger.error(f"Error sending heartbeat: {str(e)}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

def save_response_to_file(response):
    """Save response to unique JSON file with timestamp"""
    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"response_{timestamp}.json"
        file_path = os.path.join(SAVE_DIR, filename)

        # Remove non-serializable data if any
        serializable_response = {k: v for k, v in response.items() if v is not None}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_response, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Response saved to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save response: {str(e)}")
        return None

async def blind_glasses_handler(websocket):
    blind_guidance_model = None
    heartbeat_task = None
    
    try:
        blind_guidance_model = BlindDetection()
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        logger.info(f'Connection established with device: {websocket.remote_address}')

        async for data in websocket:
            try:
                message = json.loads(data)
                logger.info(f"Received message from {websocket.remote_address}")

                image_base64 = message.get("image")
                if not image_base64:
                    error_msg = "No image data received"
                    logger.warning(error_msg)
                    await websocket.send(json.dumps({"error": error_msg}))
                    continue

                image_bytes = base64.b64decode(image_base64)
                mode = message.get("mode", "default")

                # Process image
                obstacles = await blind_guidance_model.image_processing(
                    image_bytes, 
                    glasses_mode=mode
                )

                # Prepare response
                response = {
                    "audio": obstacles.audio_bytes,
                    'text_command': obstacles.answer,
                    'imp_image_info': obstacles.imp_image_info,
                    'other_image_info': obstacles.other_image_info,
                    'resized_image': obstacles.resized_image,
                    'mode_selection': obstacles.mode_selection,
                    'medicine_info': obstacles.medicine_info
                }

                # Log response details
                logger.info(f'Medicine info: {obstacles.medicine_info}')
                logger.info(f'Text command: {obstacles.answer}')
                logger.info(f'Important image info: {obstacles.imp_image_info}')
                
                # Save response to file
                save_response_to_file(response)

                # Send response back
                await websocket.send(json.dumps(response))
                logger.info("Response sent to client")

            except json.JSONDecodeError:
                error_msg = "Invalid JSON format"
                logger.error(error_msg)
                await websocket.send(json.dumps({"error": error_msg}))
            except base64.binascii.Error:
                error_msg = "Invalid base64 image data"
                logger.error(error_msg)
                await websocket.send(json.dumps({"error": error_msg}))
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(error_msg, exc_info=True)
                await websocket.send(json.dumps({"error": error_msg}))

    except Exception as e:
        logger.error(f"Connection error: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Closing connection with {websocket.remote_address}")
        if heartbeat_task:
            heartbeat_task.cancel()
        # Add any cleanup code for the model if needed
        if blind_guidance_model:
            try:
                # If your model has a cleanup method
                await blind_guidance_model.cleanup()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {str(e)}")

# Add code to start the server if needed
async def main():
    async with websockets.serve(blind_glasses_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
