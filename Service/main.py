import uvicorn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Service.api import create_app, register_router

app = create_app()

register_router(app)

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8888)