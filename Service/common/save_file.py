import os
import json

from datetime import datetime
from Service.logs.log import logger


def save_response_to_file(response):
    """Save response to unique JSON file with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"response_{timestamp}.json"
        file_path = os.path.join(file_path, filename)

        serializable_response = {k: v for k, v in response.items() if v is not None}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_response, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Response saved to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save response: {str(e)}")
        return None
