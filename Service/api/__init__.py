from fastapi import FastAPI
from .app import router as blind_detection 

base_api_url = "/algorithm/api"
def create_app():
    app = FastAPI()
    return app

def register_router(app: FastAPI):
    app.include_router(router = blind_detection, prefix = base_api_url)