import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM API ---
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_model_name: str = Field(default="deepseek-chat", alias="LLM_MODEL_NAME")

    # --- Embedding API ---
    embedding_api_url: str = Field(default="http://localhost:8002/v1", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="EMPTY", alias="EMBEDDING_API_KEY")
    embedding_model_name: str = Field(default="QwenEmbedding", alias="EMBEDDING_MODEL_NAME")
    embedding_dim: int = Field(default=2560, alias="EMBEDDING_DIM")

    # --- Reranker API ---
    rerank_api_url: str = Field(default="http://localhost:8001/v1", alias="RERANK_API_URL")
    rerank_api_key: str = Field(default="EMPTY", alias="RERANK_API_KEY")
    rerank_model_name: str = Field(default="QwenReranker", alias="RERANK_MODEL_NAME")

    # --- FIGO vLLM ---
    figo_vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="FIGO_VLLM_BASE_URL")
    figo_vllm_model_name: str = Field(default="OriClinical", alias="FIGO_VLLM_MODEL_NAME")
    figo_fallback_enabled: bool = Field(default=True, alias="FIGO_FALLBACK_ENABLED")

    # --- Databases ---
    neo4j_uri: str = Field(default="neo4j://localhost:7688", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")

    # --- Model Paths ---
    moe_model_path: str = Field(default="./result/moe_router.pth", alias="MOE_MODEL_PATH")
    attention_model_path: str = Field(default="./result/clinical_attention_v3.pth", alias="ATTENTION_MODEL_PATH")
    vdb_entities_path: str = Field(
        default="./result/vdb_entities.json",
        alias="VDB_ENTITIES_PATH",
    )
    working_dir: str = Field(default="./pathorag_core/working", alias="WORKING_DIR")
    cluster_rag_dir: str = Field(default="./CLUSTER_RAG_Endometrial", alias="CLUSTER_RAG_DIR")

    # --- Server ---
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # --- Adaptive Top-K ---
    adaptive_k_min: int = Field(default=5, alias="ADAPTIVE_K_MIN")
    adaptive_k_max: int = Field(default=20, alias="ADAPTIVE_K_MAX")
    adaptive_k_drop_threshold: float = Field(default=0.5, alias="ADAPTIVE_K_DROP_THRESHOLD")

    # --- Cross-Lingual Retrieval ---
    en_retrieval_enabled: bool = Field(default=True, alias="EN_RETRIEVAL_ENABLED")
    en_retrieval_top_k: int = Field(default=30, alias="EN_RETRIEVAL_TOP_K")

    # --- XGBoost Thresholds ---
    radiotherapy_threshold: float = 0.5
    chemotherapy_threshold: float = 0.5
    targeted_therapy_threshold: float = 0.2
    immunotherapy_threshold: float = 0.2
    hormone_therapy_threshold: float = 0.3

    @property
    def thresholds(self) -> dict:
        return {
            "radiotherapy": self.radiotherapy_threshold,
            "chemotherapy": self.chemotherapy_threshold,
            "targeted_therapy": self.targeted_therapy_threshold,
            "immunotherapy": self.immunotherapy_threshold,
            "hormone_therapy": self.hormone_therapy_threshold,
        }

    @property
    def nccn_reference_path(self) -> Path:
        p = Path(self.cluster_rag_dir) / "references" / "NCCN_2026v1.md"
        return p.resolve()

    @property
    def v4_model_dir(self) -> Path:
        return Path(self.cluster_rag_dir) / "models"


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
