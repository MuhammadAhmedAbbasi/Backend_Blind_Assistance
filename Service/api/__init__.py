from fastapi import FastAPI
from .app import router as blind_detection 
from .drug_detection_api import drug_router

base_api_url = "/algorithm/api"
def create_app(lifespan=None):
    app = FastAPI(lifespan=lifespan)
    return app

def register_router(app: FastAPI):
    app.include_router(router = blind_detection, prefix = base_api_url)
    app.include_router(router = drug_router, prefix = base_api_url)
