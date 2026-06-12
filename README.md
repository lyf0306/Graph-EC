# PathoRAG — Two-Stage CDSS for Endometrial Cancer

A clinical decision support system that combines **fine-tuned LLM staging** (PathoLLM) with **hypergraph-augmented retrieval** (PathoRAG) for evidence-based treatment recommendations in endometrial cancer.

## Architecture

```
Stage 1: PathoLLM                    Stage 2: PathoRAG
(fine-tuned FIGO staging)            (KG + HGNN + MoE CDSS)

Pathology Report                     Patient Record + FIGO Stage
      │                                      │
      ▼                                      ▼
DeepSeek-R1-32B + LoRA               KG Retrieval ←→ Vector Retrieval
      │                                      │
      ▼                                      ▼
FIGO 2023 Stage + Reasoning          MoE Dynamic Fusion (ARF Gate)
                                         │
                                         ▼
                                    HGNN Evidence Scoring
                                         │
                                         ▼
                                    Comorbidity Agent Screening
                                         │
                                         ▼
                                    MDT Report (ESGO Risk + Treatment)
```

## Key Components

| Component | File | Role |
|-----------|------|------|
| **PathoRAG Engine** | `pathorag_core/pathorag.py` | Hybrid graph+vector retrieval with coherence scoring |
| **HGNN Scorer** | `pathorag_core/hyper_attention.py` | Multi-head hypergraph attention for clinical evidence ranking |
| **ARF Gate** | `training/train_arf_gate.py` | 3-layer MLP (960→256→64→1) dynamic fusion of graph/vector streams |
| **CAES** | `training/train_caes.py` | Cross-attention entity scorer with pairwise ranking loss |
| **Entity Extraction** | `pathorag_core/operate.py` | 5-role semantic entity extraction from clinical text |
| **REST API** | `api/` | FastAPI server with 9-stage MDT pipeline |
| **Agent** | `agent/` | Comorbidity screening & treatment safety agent |

## Data Pipeline

```
Raw Patient Records (Word/PDF)
    │  data/extract_patients.py: DeepSeek API extracts structured features
    ▼
clinical_dataset_v3.jsonl
    │  data/build_training_data.py: 3-stage neuro-symbolic refinement
    │    (i)  Keyword pre-filter
    │    (ii) LLM Judge QC (DeepSeek, temperature=0.0)
    │    (iii) Graph score threshold (pos_graph ≥ 0.1)
    ▼
moe_training_data_v2_subgraph.json  (ARF Gate + CAES training data)
    │
    ├──► training/train_arf_gate.py: Train ARF Gate (BCE loss, real-valued graph scores)
    └──► training/train_caes.py: Train CAES (pairwise ranking loss)
```

### Negative Mining (2 KG Rules)

1. **FIGO stage mismatch** — wrong stage → wrong treatment
2. **Contraindication induction** — triggered only for complex cases with high-risk comorbidities

### Neuro-Symbolic Graph Scoring

Real-valued scores from PathoRAG hybrid inference (`mode="hybrid"`, top_k=50):
```
score = coherence × sim^0.5
```
where `sim` = Reranker semantic similarity (> 0.4 threshold, min 0.01).

## Quick Start

### Prerequisites

- Python 3.10+
- Neo4j (graph database)
- Milvus (vector database)
- QwenEmbedding (vLLM on port 8002)
- QwenReranker (vLLM on port 8001)
- LLM API key (OpenAI-compatible: DeepSeek / Qwen / GPT / etc.)

### Setup

```bash
cd PathoRAG

# Install dependencies
pip install -r requirements-web.txt
pip install -r requirements-observability.txt

# Core dependencies
pip install neo4j pymilvus sentence-transformers openai torch

# Configure
cp .env.example .env
# Edit .env with your API keys and database URIs

# Verify
python -c "from pathorag_core import PathoRAG, QueryParam; print('OK')"
```

### Start API Server

```bash
cd api
python main.py
# → http://localhost:8000
# → /health — health check
# → /analyze — full MDT pipeline
# → /docs — Swagger UI
```

## Key Files

### Training Data Construction
- `data/extract_patients.py` — Extract structured patient features from raw records
- `data/build_training_data.py` — Neuro-symbolic refinement pipeline for ARF Gate + CAES data
- `data/build_moe_trainset.py` — MoE training set builder
- `data/compute_idf.py` — Entity-role IDF computation

### Model Training
- `training/train_arf_gate.py` — ARF Gate (MoE router) training
- `training/train_caes.py` — CAES (Cross-Attention Evidence Scorer) training
- `training/train_scorer.py` — Evidence scorer training

### Testing
- `tests/test_pipeline.py` — Main integration test (full pipeline)
- `tests/test_workflow.py` — Workflow validation
- `tests/test_mlp.py` — MLP component unit test

### API
- `api/main.py` — FastAPI application entry point
- `api/config.py` — Configuration (pydantic-settings from .env)
- `api/services/pipeline_service.py` — 9-stage MDT pipeline orchestrator
- `api/services/resource_manager.py` — PathoRAG + CAES resource lifecycle
- `api/services/patient_retriever.py` — KNN + XGBoost patient similarity

## ESGO 2025 Risk Classification

PathoRAG implements the full ESGO 2025 decision tree:
- Low / Intermediate / High-intermediate / High / Advanced / Metastatic
- Integrates: FIGO stage, histology, grade, LVSI, myometrial invasion, molecular subtype

## MDT Report (9-Stage Pipeline)

1. Patient profile extraction (LLM)
2. Structured feature extraction (40+ fields)
3. ESGO 2025 risk classification
4. Knowledge graph query (graph + vector)
5. MoE dynamic weight computation
6. KG evidence retrieval + HGNN ranking
7. Rerank + reference formatting
8. Similar patient retrieval + treatment prediction (KNN + XGBoost)
9. Comorbidity screening agent → Final MDT report

## Observability

Optional OpenTelemetry + Prometheus instrumentation:
```python
from pathorag_core.instrumentation import init_instrumentation
init_instrumentation(service_name="pathorag")
```

Metrics exported:
- PathoRAG query latency histograms
- ARF Gate fusion weight α distribution
- CAES attention score distribution
- Pipeline stage duration tracking

## License

Proprietary — research use only.
