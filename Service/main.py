import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from websockets.asyncio.server import serve

# Allow both `python -m Service.main` and `python Service/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Service.api import create_app, register_router
from Service.handler.handler_websocket import blind_glasses_handler


@asynccontextmanager
async def lifespan(_app):
    async with serve(
        blind_glasses_handler,
        "0.0.0.0",
        8766,
        max_size=10 * 1024 * 1024,
    ):
        yield


app = create_app(lifespan=lifespan)
register_router(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
