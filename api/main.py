import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_config
from api.services.resource_manager import ResourceManager
from api.services.pipeline_service import PipelineStageError
from api.middleware.error_handler import (
    pipeline_stage_error_handler,
    timeout_error_handler,
    generic_exception_handler,
)
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import health, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    rm = ResourceManager(config)
    await rm.initialize()
    app.state.resources = rm
    yield
    await rm.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PathoRAG API",
        version="1.0.0",
        description="Endometrial cancer clinical decision support system with knowledge graph RAG",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://10.91.11.250:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RateLimitMiddleware)

    app.add_exception_handler(PipelineStageError, pipeline_stage_error_handler)
    app.add_exception_handler(asyncio.TimeoutError, timeout_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(health.router)
    app.include_router(analyze.router)

    return app


app = create_app()
