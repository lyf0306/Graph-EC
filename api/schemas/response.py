from typing import Optional
from pydantic import BaseModel, Field


class EvidenceFragment(BaseModel):
    content: str
    source_ids: list[str] = []
    final_score: float = 0.0
    evidence_type: str = ""


class BibliographyEntry(BaseModel):
    ref_index: int
    pmid: str = ""
    title: str = ""
    guidelines: str = ""
    source_id: str = ""


class SimilarPatient(BaseModel):
    patient_id: str = ""
    distance: float = 0.0
    description: str = ""
    historical_treatment: str = ""


class ComorbidityScreeningResult(BaseModel):
    safety_warnings: list[str] = []
    deescalation_advice: list[str] = []
    monitoring_recommendations: list[str] = []
    coverage_summary: dict = {}
    reference_log: list[dict] = []


class AnalyzeResponse(BaseModel):
    request_id: str = ""
    patient_profile_md: str = ""
    esgo_risk_classification: str = ""
    esgo_raw_label: str = ""
    knowledge_evidence: list[EvidenceFragment] = []
    bibliography: list[BibliographyEntry] = []
    similar_patients: list[SimilarPatient] = []
    xgb_predictions: dict[str, int] = {}
    comorbidity_screening: ComorbidityScreeningResult = Field(default_factory=ComorbidityScreeningResult)
    mdt_report: str = ""
    mdt_overrides: list[str] = []


class ProfileResponse(BaseModel):
    patient_profile_md: str = ""
    bilingual_keywords: str = ""
    new_patient_dict: dict = {}
    esgo_risk_classification: str = ""
    figo_stage: str = ""


class FigoResponse(BaseModel):
    figo_stage: Optional[str] = Field(
        None,
        description="提取到的 FIGO 分期字符串，如 'IIIC1ii'；无法提取时为 null",
    )
    raw_output: str = Field(
        "",
        description="模型完整原始输出，供调试或前端展示推理过程",
    )
    model_used: str = Field(
        "",
        description="实际使用的模型名称，如 'OriClinical' 或 'deepseek-chat'",
    )


class HealthResponse(BaseModel):
    status: str
    neo4j: bool = False
    milvus: bool = False
    embedding_api: bool = False


class ErrorResponse(BaseModel):
    error: str
    stage: str = ""
    detail: str = ""
