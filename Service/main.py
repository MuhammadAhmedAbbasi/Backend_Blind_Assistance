import uvicorn
import sys
import os
import asyncio
from websockets.asyncio.server import serve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Service.api import create_app, register_router
from Service.handler.handler_websocket import blind_glasses_handler


app = create_app()
register_router(app)

async def websocket_api():
    async with serve(blind_glasses_handler, '0.0.0.0', 8765,max_size=None):
        await asyncio.Future()

async def start_fastapi_service():
    """Start the FastAPI server."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8888)
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    await asyncio.gather(
        websocket_api(),
        start_fastapi_service()
    )

if __name__ == "__main__":
    asyncio.run(main())