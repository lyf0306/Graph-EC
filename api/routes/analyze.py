import json
import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

from api.schemas.request import AnalyzeRequest, ProfileRequest, FigoRequest
from api.schemas.response import AnalyzeResponse, ProfileResponse, FigoResponse, ErrorResponse
from api.dependencies import get_pipeline_service
from api.services.pipeline_service import ESGO_MAPPING

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        200: {"description": "Full pipeline result"},
        502: {"model": ErrorResponse, "description": "Pipeline stage failed"},
        504: {"model": ErrorResponse, "description": "Upstream timeout"},
    },
)
async def analyze(request: AnalyzeRequest, req: Request):
    service = get_pipeline_service(req)
    timeout = 300.0
    result = await asyncio.wait_for(
        service.execute(request.patient_case, request.top_k_similar),
        timeout=timeout,
    )
    return AnalyzeResponse(**result)


@router.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest, req: Request):
    service = get_pipeline_service(req)

    STREAM_TIMEOUT = 600.0

    async def event_generator():
        try:
            ait = _event_iter(service, request).__aiter__()
            while True:
                try:
                    event_type, payload = await asyncio.wait_for(
                        ait.__anext__(), timeout=STREAM_TIMEOUT
                    )
                except StopAsyncIteration:
                    break
                if event_type == "token":
                    yield {"event": "token", "data": payload}
                elif event_type == "error":
                    yield {"event": "error", "data": json.dumps(payload, ensure_ascii=False)}
                else:
                    yield {
                        "event": event_type,
                        "data": json.dumps(payload, ensure_ascii=False, default=str),
                    }
        except asyncio.TimeoutError:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"stage": "timeout", "message": f"Stream timed out after {STREAM_TIMEOUT}s"},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())

async def _event_iter(service, request):
    async for event_type, payload in service.execute_stream(
        request.patient_case, request.top_k_similar
    ):
        yield event_type, payload


@router.post("/analyze/profile-only", response_model=ProfileResponse)
async def profile_only(request: ProfileRequest, req: Request):
    service = get_pipeline_service(req)
    profile_md, bilingual_keywords = await service._extract_patient_profile(request.patient_case)
    patient_dict = await service._get_structured_features(request.patient_case)
    esgo_raw = patient_dict.get("esgo_risk_group", "unknown")
    esgo_level, _ = ESGO_MAPPING.get(esgo_raw.lower(), ("未知", "Unknown"))
    return ProfileResponse(
        patient_profile_md=profile_md,
        bilingual_keywords=bilingual_keywords,
        new_patient_dict=patient_dict,
        esgo_risk_classification=esgo_level,
        figo_stage=patient_dict.get("stage_raw", ""),
    )


# ─── POST /api/v1/figo-stage ─────────────────────────────────────────────────

@router.post(
    "/figo-stage",
    response_model=FigoResponse,
    responses={
        200: {"description": "FIGO 分期结果"},
        503: {"model": ErrorResponse, "description": "vLLM 服务不可用"},
        500: {"model": ErrorResponse, "description": "分期推断内部错误"},
    },
)
async def figo_stage(request: FigoRequest, req: Request):
    """
    调用 OriClinical vLLM 对病理报告进行 FIGO 分期。
    先于 profile-only 调用（前端串行编排）。
    """
    figo_svc = getattr(req.app.state.resources, "figo_service", None)
    if figo_svc is None:
        logger.error("/figo-stage: FigoService 未挂载到 app.state.resources")
        return JSONResponse(
            status_code=503,
            content={"error": "FigoService 未初始化", "stage": "figo", "detail": ""},
        )

    try:
        result = await asyncio.wait_for(
            figo_svc.predict(request.patient_case),
            timeout=620.0,   # 略大于 vLLM request_timeout=600
        )
    except asyncio.TimeoutError:
        logger.error("/figo-stage: 调用 vLLM 超时")
        return JSONResponse(
            status_code=503,
            content={"error": "vLLM 服务超时", "stage": "figo", "detail": "request_timeout exceeded"},
        )
    except Exception as exc:
        logger.exception("/figo-stage: 未预期异常 %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "FIGO 分期推断失败", "stage": "figo", "detail": str(exc)},
        )

    return FigoResponse(**result)
