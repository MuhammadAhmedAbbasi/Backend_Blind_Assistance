from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi import APIRouter
import logging
from Service.model_service.detection_new import blind_algo
router = APIRouter(
    prefix = "/blind",
    tags = ["blind"]
)
logger = logging.getLogger(__name__)


@router.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...),
    glasses_mode: str  = Form(...)
):
    try:
        print(glasses_mode)
        image_bytes = await file.read()
        obstacles = await blind_algo.image_processing(
            image_bytes=image_bytes,
            glasses_mode=glasses_mode,
        )
        return JSONResponse(content={
            "audio": obstacles.audio_bytes, 
            'text_command': obstacles.answer, 
            'imp_image_info': obstacles.imp_image_info, 
            'other_image_info': obstacles.other_image_info,
            'resized_image': obstacles.resized_image,
            'mode_selection': obstacles.mode_selection,
            'medicine_info': obstacles.medicine_info})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("Image detection failed")
        raise HTTPException(status_code=500, detail="Image processing failed")
