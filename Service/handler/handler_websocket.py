from Service.model_service.detection_new import BlindDetection
import json
import base64
import websockets
import asyncio
import os

import logging
from collections import defaultdict, deque
import time
from Service.config import audio_save_path, file_save_path
from Service.logs.log import logger


HEARTBEAT_INTERVAL = 5
blind_guidance_model = BlindDetection()
base_path = (os.path.dirname(os.path.abspath(__name__)))
async def initialize_model():
    await blind_guidance_model.image_processing(
            image_path = os.path.join(base_path, "save_path/startup_save_path/detection.jpg"),  # 3. Use / instead of \
            glasses_mode = 'detection'
        )
    await blind_guidance_model.image_processing(
            image_path = os.path.join(base_path, "save_path/startup_save_path/drug.jpg"),  # 3. Use / instead of \
            glasses_mode = 'drug_detection'
        )
    logger.info(f'The detection and drug check has completed')

# 2. Run the async initialization before starting the server
asyncio.run(initialize_model())


base_path = (os.path.dirname(os.path.abspath(__name__)))

file_path = os.path.join(base_path, file_save_path)
audio_path = os.path.join(base_path, audio_save_path)

# Ensure save directories exist
os.makedirs(file_path, exist_ok=True)
os.makedirs(audio_path, exist_ok=True) 

# Per-client message queues
client_queues = defaultdict(deque)

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


async def blind_glasses_handler(websocket):

    heartbeat_task = None
    client_id = id(websocket)
    
    count = 1

    try:
        
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        logger.info(f'Connection established with device: {websocket.remote_address}')
        start_t = time.time()
        # if count == 1:
        #     _ = await blind_guidance_model.image_processing(
        #             image_path = os.path.join(base_path, "save_path\startup_save_path\detection.jpg"),
        #             glasses_mode = 'detection'
        #         )
        #     _ = await blind_guidance_model.image_processing(
        #             image_path = os.path.join(base_path, "save_path\startup_save_path\drug.jpg"),
        #             glasses_mode = 'drug_detection'
        #         )
            # logging.info(f'The detection and drug check has completed')
            # count += 1
            
        async for data in websocket:
            try:
                 
                message = json.loads(data)
                logger.info(f"Received message from {websocket.remote_address}")
                logger.info(f'The client id is {client_id}')
                end_t = time.time() - start_t
                logger.info(f"The first message time is : {end_t}")
                main_received_data = time.time()
                # Extract mode early
                mode = message.get("mode", "default")
                logger.info(f"The mode received is : {mode}")

                # Append the incoming message once
                client_queues[client_id].append(message)

                logger.info(f"Total messages in clients are: {len(client_queues[client_id])}")
                last_mode = client_queues[client_id][-2].get("mode", "default") if len(client_queues[client_id]) > 1 else "default"
                current_mode = client_queues[client_id][-1].get("mode", "default")
                if last_mode != current_mode:
                    # Clear queue except last message
                    last_msg = client_queues[client_id].pop()
                    client_queues[client_id].clear()
                    logger.info(f'After clearing the length of queue is : {len(client_queues[client_id])}')
                    client_queues[client_id].append(last_msg)

                # Process the oldest message (this removes it from the queue)
                raw_message = client_queues[client_id].popleft()

                # Decode image bytes from raw message
                image_base64 = raw_message.get("image")
                if not image_base64:
                    error_msg = "No image data received"
                    logger.warning(error_msg)
                    await websocket.send(json.dumps({"error": error_msg}))
                    continue

                image_bytes = base64.b64decode(image_base64)
                raw_mode = raw_message.get("mode", "default")
                start_time = time.time()
                # Process image using the model
                obstacles = await blind_guidance_model.image_processing(
                    image_bytes = image_bytes,
                    glasses_mode=raw_mode
                )
                end_time = time.time() - start_time
                logger.info(f'The time taken for response is: {end_time}')
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
                end_overall = time.time() - main_received_data
                logger.info(f'The overall time of processing: {end_overall}')
                # 保存音频文件
                
                if obstacles.audio_bytes is not None:
                    logger.info(f'The audio check: {obstacles.audio_bytes[:10]}')
                    # 调用音频保存函数
                    # save_audio_file(obstacles.audio_bytes, client_id)
                else:
                    logger.info(f'The audio check: {obstacles.audio_bytes}')
                # save_response_to_file(response)
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
        client_queues.pop(client_id, None)
        if blind_guidance_model:
            try:
                await blind_guidance_model.cleanup()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {str(e)}")
