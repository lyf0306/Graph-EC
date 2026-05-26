from pydantic import BaseModel, Field, ConfigDict


class AnalyzeRequest(BaseModel):
    patient_case: str = Field(
        ...,
        min_length=10,
        max_length=80000,
        description="Raw clinical case text (Chinese medical record)",
    )
    top_k_similar: int = Field(default=3, ge=1, le=10, description="Number of similar patients to retrieve")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_case": "# 患者病情：\n## 现病史:\n患者绝经...",
                "top_k_similar": 3,
            }
        }
    )


class ProfileRequest(BaseModel):
    patient_case: str = Field(..., min_length=10, max_length=80000)


class FigoRequest(BaseModel):
    patient_case: str = Field(
        ...,
        min_length=10,
        max_length=80000,
        description="原始病理报告文本（中文）",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_case": "一、全子宫：1.子宫内膜样腺癌，Ⅱ级，浸润深肌层..."
            }
        }
    )
