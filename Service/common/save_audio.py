from datetime import datetime
import os
import base64
from Service.logs.log import logger

def save_audio_file(audio_bytes, client_id, AUDIO_SAVE_DIR):
    """保存音频字节到文件"""
    try:
        if not audio_bytes:
            logger.warning("没有音频数据可保存")
            return None
            
        # 生成唯一的文件名，包含时间戳和客户端ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"audio_{client_id}_{timestamp}.wav"  # 假设音频是WAV格式
        file_path = os.path.join(AUDIO_SAVE_DIR, filename)
        
        # 解码base64音频并保存
        decoded_audio = base64.b64decode(audio_bytes)
        with open(file_path, 'wb') as f:
            f.write(decoded_audio)
        
        logger.info(f"音频文件已保存至: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"保存音频文件时出错: {str(e)}")
        return None