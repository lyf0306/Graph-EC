from fastapi import Request

from api.services.resource_manager import ResourceManager
from api.services.pipeline_service import PipelineService


def get_resource_manager(request: Request) -> ResourceManager:
    return request.app.state.resources


def get_pipeline_service(request: Request) -> PipelineService:
    rm = request.app.state.resources
    return PipelineService(rm)
