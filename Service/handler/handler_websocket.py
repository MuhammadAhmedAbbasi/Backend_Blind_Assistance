from Service.model_service.detection_new import blind_algo
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
from datetime import datetime


HEARTBEAT_INTERVAL = 5


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file_path = os.path.join(base_path, file_save_path)
audio_path = os.path.join(base_path, audio_save_path)

# Ensure save directories exist
os.makedirs(file_path, exist_ok=True)
os.makedirs(audio_path, exist_ok=True) 

# Per-client message queues
client_queues = defaultdict(lambda: deque(maxlen=1))

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
            logger.info(" Heartbeat sent (心跳包已发送)")

        except websockets.exceptions.ConnectionClosed:
            logger.info(" Connection closed during heartbeat (心跳时连接已关闭)")
            break
        except Exception as e:
            logger.error(f" Error sending heartbeat (发送心跳错误): {str(e)}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def blind_glasses_handler(websocket):
    heartbeat_task = None
    client_id = id(websocket)
    
    count = 1

    try:
        
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        logger.info(f" Connection established with device {websocket.remote_address} (已与设备建立连接)")
            
        async for data in websocket:
            try:
                my_own_start_T = time.time()
                message = json.loads(data)
                logger.info(f" Message received from {websocket.remote_address} (收到消息) | client_id={client_id}")
                start_t = message.get("send_time")
                if start_t:
                    try:
                        start_t = float(start_t)

                        # Convert ms → seconds if too large
                        if start_t > 1e12:  
                            start_t /= 1000.0

                        latency_up = time.time() - start_t
                        logger.info(
                            f" Client→Server latency: {latency_up:.3f}s | 客户端到服务器延迟: {latency_up:.3f}秒"
                        )
                    except Exception as e:
                        logger.warning(f" Could not parse send_time (无法解析 send_time): {e}")
                else:
                    logger.warning(" No send_time in message; cannot compute latency (消息缺少 send_time，无法计算延迟)")
                parsing_duration = time.time() - my_own_start_T
                logger.info(f" Server parsing time: {parsing_duration:.3f}s | 服务器解析时间: {parsing_duration:.3f}秒")

                code_processing_time = time.time()
                # Extract mode early
                mode = message.get("mode", "default")
                logger.info(f"The mode received is : {mode}")

                # Append the incoming message once
                # Keep only the newest frame for live guidance; stale frames are dropped.
                client_queues[client_id].clear()
                client_queues[client_id].append(message)

                logger.info(f"Total messages in clients are: {len(client_queues[client_id])}")
                # Process the newest message.
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
                models_processing_start_time = time.time()
                # Process image using the model
                obstacles = await blind_algo.image_processing(
                    image_bytes = image_bytes,
                    glasses_mode=raw_mode
                )
                models_processing_end_time = time.time() - models_processing_start_time
                logger.info(f" Model processing time: {models_processing_end_time:.3f}s | 模型处理时间: {models_processing_end_time:.3f}秒")
                gaojinwei_time = str(datetime.now())[11:23]
                logger.info(f"The time I sent Glasses is : {gaojinwei_time}")
                # Prepare response
                response = {
                    "audio": obstacles.audio_bytes,
                    'text_command': obstacles.answer,
                    'imp_image_info': obstacles.imp_image_info,
                    'other_image_info': obstacles.other_image_info,
                    'resized_image': obstacles.resized_image,
                    'mode_selection': obstacles.mode_selection,
                    'medicine_info': obstacles.medicine_info,
                    "sending_time": gaojinwei_time
                }
                
                # 保存音频文件
                send_parse_time = time.time()
                if obstacles.audio_bytes is not None:
                    logger.info(f"🎵 Audio check OK (音频检查成功): {obstacles.audio_bytes[:10]}")
                    # 调用音频保存函数
                    # save_audio_file(obstacles.audio_bytes, client_id)
                else:
                    logger.info("🎵 No audio generated (未生成音频)")
                # save_response_to_file(response)
                await websocket.send(json.dumps(response))
                logger.info(" Response sent to client (响应已发送给客户端)")
                end_overall = time.time() - code_processing_time
                logger.info(f'The overall time of processing (Models + Sending) 处理的总体时间（模型 + 发送: {end_overall}')
                logger.info(f'The Only sending time is 唯一发送时间是: {time.time() - send_parse_time}')
                

            except json.JSONDecodeError:
                error_msg = "Invalid JSON format"
                logger.error(error_msg)
                await websocket.send(json.dumps({"error": error_msg}))
            except base64.binascii.Error:
                error_msg = "Invalid base64 image data"
                logger.error(error_msg)
                await websocket.send(json.dumps({"error": error_msg}))
            except Exception as e:
                logger.error("Error processing request", exc_info=True)
                await websocket.send(json.dumps({"error": "Image processing failed"}))

    except Exception as e:
        logger.error(f"Connection error: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Closing connection with {websocket.remote_address}")
        if heartbeat_task:
            heartbeat_task.cancel()
        client_queues.pop(client_id, None)
