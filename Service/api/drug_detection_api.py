from fastapi import UploadFile, File, APIRouter, HTTPException
from Service.model_service.detection_new import blind_algo

drug_router = APIRouter(prefix = "/drug_detection",
                        tags = ['drug_detection']
                        )

@drug_router.post("/detect")
async def drug_api(
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        
        result = await blind_algo.image_processing(
            image_bytes=image_bytes,
            glasses_mode="Drug_detection",
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Drug detection failed") from exc
