from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from fastapi import APIRouter
from Service.model_service.detection import image_processing
from pydantic import BaseModel
from typing import Optional
router = APIRouter(
    prefix = "/blind",
    tags = ["blind"]
)


@router.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        obstacles = image_processing(image_bytes)
        print(f'The type is:{type(obstacles.resized_image)}')
        
        return JSONResponse(content={
            "audio": obstacles.audio_bytes, 
            'text_command': obstacles.answer, 
            'imp_image_info': obstacles.imp_image_info, 
            'other_image_info': obstacles.other_image_info,
            'resized_image': obstacles.resized_image
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))