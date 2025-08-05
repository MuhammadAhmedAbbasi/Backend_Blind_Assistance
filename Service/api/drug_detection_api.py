from fastapi import UploadFile, File, APIRouter

drug_router = APIRouter(prefix = "/drug_detection",
                        tags = ['drug_detection']
                        )

@drug_router.post("/detect")
async def drug_api(
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        
        pass
    except:
        pass