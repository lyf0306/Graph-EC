import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse

from api.services.pipeline_service import PipelineStageError


async def pipeline_stage_error_handler(request: Request, exc: PipelineStageError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "pipeline_stage_failed",
            "stage": exc.stage,
            "detail": exc.message,
        },
    )


async def timeout_error_handler(request: Request, exc: asyncio.TimeoutError):
    return JSONResponse(
        status_code=504,
        content={
            "error": "upstream_timeout",
            "detail": "A downstream service (LLM, Neo4j, Milvus) did not respond in time.",
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc),
        },
    )
