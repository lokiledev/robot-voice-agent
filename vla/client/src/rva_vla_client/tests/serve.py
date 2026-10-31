import asyncio
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import AnyHttpUrl


@asynccontextmanager
async def serve_locally(app: FastAPI) -> AsyncGenerator[AnyHttpUrl]:
    """Serve `app` on a free localhost port for the duration of the context."""
    # Listening before uvicorn starts queues early connections, so no need to wait for startup.
    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        yield AnyHttpUrl(f"http://127.0.0.1:{port}")
    finally:
        server.should_exit = True
        await serving
