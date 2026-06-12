# PathoRAG Migration Guide

## Quick Reference

| Item | Old | New |
|------|-----|-----|
| Class name | `GraphR1` | `PathoRAG` |
| File name | `graphr1.py` | `pathorag.py` |
| Proto file | `graphr1_proto.py` | `pathorag_proto.py` |
| Python import | `from graphr1 import ...` | `from pathorag_core import ...` |
| Logger namespace | `graphr1` / `graphr1.instrumentation` | `pathorag_core` / `pathorag_core.instrumentation` |
| Tracer name | `"graphr1"` | `"pathorag_core"` |
| Cache directory | `graphr1_cache_*` | `pathorag_core_cache_*` |

## Preserved Identifiers (DO NOT CHANGE)

These are database-level identifiers in Oracle and must remain as-is:

- `graphr1_graph` — Oracle property graph name
- `graphr1_graph_nodes` — Oracle node table
- `graphr1_graph_edges` — Oracle edge table
- `graphr1_doc_chunks` — Oracle chunk storage table

## Files Updated (21 files)

### External imports (7 files)
- `data/build_training_data.py`
- `api/services/resource_manager.py`
- `api/services/pipeline_service.py`
- `tests/test_workflow.py`
- `tests/test_mlp.py`
- `tests/test_pipeline.py`
- `data/build_moe_trainset.py`

### Internal imports (8 files in pathorag_core/)
- `base.py`
- `graphr1.py` (cache name only; renamed to `pathorag.py`)
- `graphr1_proto.py` (renamed to `pathorag_proto.py`)
- `instrumentation.py`
- `utils.py`
- `hyper_attention.py`
- `kg/chroma_impl.py`
- `kg/milvus_impl.py`
- `kg/mongo_impl.py`
- `kg/tidb_impl.py`

## Dependencies

### Python
- Python >= 3.10

### Core
- `neo4j` (AsyncGraphDatabase)
- `pymilvus` (Milvus vector DB)
- `sentence-transformers` (embedding + reranker)
- `openai` (AsyncOpenAI — OpenAI-compatible API)
- `torch` (MoE router + CAES attention)

### API Layer
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.0.0
pydantic-settings>=2.5.0
```

### Observability (optional)
```
opentelemetry-api>=1.27.0
opentelemetry-sdk>=1.27.0
opentelemetry-exporter-otlp-proto-grpc>=1.27.0
prometheus-client>=0.21.0
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_API_KEY` | LLM API key (OpenAI-compatible) | — |
| `LLM_BASE_URL` | LLM API endpoint | `https://api.deepseek.com` (example) |
| `LLM_MODEL_NAME` | Model name | `deepseek-chat` (example) |
| `EMBEDDING_API_URL` | Embedding service endpoint | `http://localhost:8002/v1` (example) |
| `EMBEDDING_DIM` | Embedding dimension | `2560` |
| `RERANK_API_URL` | Reranker service endpoint | `http://localhost:8001/v1` (example) |
| `NEO4J_URI` | Neo4j graph DB URI | `neo4j://localhost:7688` |
| `MILVUS_URI` | Milvus vector DB URI | `http://localhost:19530` |
| `MOE_MODEL_PATH` | MoE router checkpoint | `/root/Model/moe_router.pth` |
| `ATTENTION_MODEL_PATH` | CAES model checkpoint | `/root/Model/clinical_attention_v3.pth` |
| `WORKING_DIR` | PathoRAG working directory | `/root/PathoRAG/expr` |
| `CLUSTER_RAG_DIR` | Cluster RAG directory | `/root/Graph-R1/CLUSTER_RAG_Endometrial` |

## External Services

| Service | Port | Purpose |
|---------|------|---------|
| Embedding (e.g., QwenEmbedding + vLLM) | 8002 | Dense embeddings |
| Reranker (e.g., QwenReranker + vLLM) | 8001 | Cross-encoder reranking |
| Neo4j | 7688 | Clinical knowledge graph storage |
| Milvus | 19530 | Vector similarity search |
| PathoRAG API | 8000 | REST API server |

## Directory Layout After Migration

```
PathoRAG/
├── pathorag_core/          # Renamed from graphr1/
│   ├── __init__.py         # Re-exports PathoRAG, QueryParam
│   ├── pathorag.py         # Core PathoRAG engine
│   ├── pathorag_proto.py   # Prototype variant
│   ├── base.py             # Base storage abstractions
│   ├── operate.py          # Entity extraction & KG operations
│   ├── hyper_attention.py  # HGNN clinical evidence scorer
│   ├── instrumentation.py  # OTel + Prometheus observability
│   ├── llm.py              # LLM interaction layer
│   ├── prompt.py           # Prompt templates
│   ├── storage.py          # Storage implementations
│   ├── utils.py            # Utilities (logger, embedding wrapper)
│   └── kg/                 # DB backends
│       ├── chroma_impl.py
│       ├── milvus_impl.py
│       ├── mongo_impl.py
│       ├── neo4j_impl.py
│       ├── oracle_impl.py  # Oracle property graph (graphr1_graph preserved)
│       └── tidb_impl.py
├── api/                    # REST API (FastAPI)
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── middleware/
├── data/                   # Data pipeline scripts
│   ├── extract_patients.py # Patient feature extraction
│   ├── build_training_data.py # Neuro-symbolic refinement pipeline
│   ├── build_moe_trainset.py  # MoE training set builder
│   └── compute_idf.py      # Entity-role IDF computation
├── training/               # Model training scripts
│   ├── train_arf_gate.py   # ARF Gate (MoE router) training
│   ├── train_caes.py       # CAES (Cross-Attention Evidence Scorer) training
│   └── train_scorer.py     # Evidence scorer training
├── tests/                  # Test files
│   ├── test_pipeline.py    # Main integration test
│   ├── test_workflow.py    # Workflow validation
│   └── test_mlp.py         # MLP component unit test
├── agent/                  # Comorbidity screening agent
├── result/                 # Training data & model artifacts
├── CLUSTER_RAG_Endometrial/# Cluster RAG pipeline
├── .env.example
├── requirements-web.txt
├── requirements-observability.txt
├── MIGRATION.md
└── README.md
```

## Post-Migration Verification

```bash
cd /path/to/PathoRAG
python -c "from pathorag_core import PathoRAG, QueryParam; print('OK')"
python -c "from pathorag_core.hyper_attention import init_attention_system; print('OK')"
python -c "from pathorag_core.utils import wrap_embedding_func_with_attrs; print('OK')"
```
