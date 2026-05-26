import asyncio
import os
import sys
import math
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
from openai import AsyncOpenAI

from pathorag_core import PathoRAG, QueryParam
from pathorag_core.utils import wrap_embedding_func_with_attrs
from pathorag_core.hyper_attention import init_attention_system

from api.services.figo_service import FigoService
from api.config import AppConfig

ML_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ml-worker-")
RERANK_SEMAPHORE = asyncio.Semaphore(12)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoERouter(nn.Module):
    def __init__(self, input_dim, temperature_init=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)
        self.ln4 = nn.LayerNorm(64)
        self.fc5 = nn.Linear(64, 1)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.drop3(x)
        x = F.relu(self.ln4(self.fc4(x)))
        T = torch.exp(self.log_temperature)
        return torch.sigmoid(self.fc5(x) / T)


def get_guideline_tier(guideline_str):
    if not guideline_str:
        return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str:
        return 1
    elif "FIGO" in g_str:
        return 2
    elif "NCCN" in g_str:
        return 3
    else:
        return 4


async def compute_rerank_score(query, doc, client, model_name):
    instruction = (
        "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan. "
        "The Query is typically in Chinese and the Document may be in Chinese or English. "
        "Cross-lingual semantic relevance is fully acceptable — judge by clinical facts, NOT by language match. "
        "A Document in English that discusses the same clinical condition, guideline, or treatment as the Chinese Query "
        "should be rated as highly relevant."
    )
    prompt = (
        f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
        f"The Query and Document may be in different languages — this is expected and acceptable. "
        f"Evaluate based on clinical semantic relevance, not language. "
        f"Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    try:
        async with RERANK_SEMAPHORE:
            response = await client.completions.create(
                model=model_name, prompt=prompt, max_tokens=1, temperature=0, logprobs=20
            )
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        true_logit, false_logit = -10.0, -10.0
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes":
                true_logit = max(true_logit, logprob)
            elif clean_token == "no":
                false_logit = max(false_logit, logprob)
        true_score, false_score = math.exp(true_logit), math.exp(false_logit)
        return 0.0 if true_score + false_score == 0 else true_score / (true_score + false_score)
    except Exception:
        return 0.0


async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, "driver"):
            return sources_info
        async with storage.driver.session() as session:
            if knowledge_text.startswith("【权威循证溯源："):
                match = re.search(r"【权威循证溯源：(.*?)】", knowledge_text)
                if match:
                    paper_name = match.group(1).strip()
                    cypher_query = """
                    MATCH (paper:Paper)
                    WHERE paper.name = $paper_name OR paper.pmid = $paper_name
                    RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                    """
                    result = await session.run(cypher_query, paper_name=paper_name)
                    records = await result.data()
                else:
                    records = []
            else:
                node_id = f"<hyperedge>{knowledge_text}"
                cypher_query = """
                MATCH (target)-[r:BELONG_TO]->(paper:Paper)
                WHERE target.name = $node_id
                RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                """
                result = await session.run(cypher_query, node_id=node_id)
                records = await result.data()

            for record in records:
                src_id = record.get("src_id", "Unknown").replace('"', "")
                raw_pmid = record.get("pmid")
                final_pmid = (
                    str(raw_pmid)
                    if (raw_pmid and len(str(raw_pmid)) < 20)
                    else src_id.replace("paper::", "")
                    if "paper::" in src_id
                    else "Unknown"
                )
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                sources_info.append({
                    "id": src_id,
                    "pmid": final_pmid,
                    "title": record.get("title") or "No Title",
                    "guidelines": gl_str,
                })
    except Exception:
        pass
    return sources_info


async def lookup_guideline_paper(session, guideline_name: str) -> dict | None:
    """Search Neo4j for the canonical Paper node of an authoritative guideline.

    This finds the guideline document itself (e.g., the actual NCCN PDF that was
    ingested), not papers that merely discuss or cite the guideline.
    """
    try:
        query = """
        MATCH (p:Paper)
        WHERE ANY(g IN p.guidelines WHERE toLower(g) CONTAINS toLower($guideline))
        RETURN p.name AS src_id, p.pmid AS pmid, p.title AS title, p.guidelines AS guidelines
        ORDER BY
            CASE WHEN p.title CONTAINS 'NCCN' OR p.title CONTAINS 'FIGO' OR p.title CONTAINS 'ESGO' THEN 0 ELSE 1 END,
            size(p.title) ASC
        LIMIT 1
        """
        result = await session.run(query, guideline=guideline_name)
        record = await result.single()
        if not record:
            return None
        raw_pmid = record.get("pmid")
        final_pmid = (
            str(raw_pmid)
            if (raw_pmid and len(str(raw_pmid)) < 20)
            else record.get("src_id", "Unknown").replace("paper::", "")
        )
        raw_gl = record.get("guidelines")
        gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "Guideline")
        return {
            "id": record.get("src_id", "Unknown").replace('"', ""),
            "pmid": final_pmid,
            "title": record.get("title") or "No Title",
            "guidelines": gl_str,
        }
    except Exception:
        return None


NON_AUTHORITATIVE_GUIDELINE_KEYWORDS = ("其他", "general evidence", "unknown", "", "other")


def is_non_authoritative_source(guidelines_str: str) -> bool:
    """Check if a paper's guidelines field indicates it's not a primary guideline source."""
    if not guidelines_str:
        return True
    gl_lower = guidelines_str.lower().strip()
    return any(kw in gl_lower for kw in NON_AUTHORITATIVE_GUIDELINE_KEYWORDS)


PROVENANCE_ANALYSIS_PROMPT = """You are a clinical evidence provenance expert. Analyze each knowledge fragment below and determine which authoritative clinical practice guideline it is actually derived from or restating.

For each fragment, choose ONE:
- "NCCN" — content restates NCCN guideline recommendations (dosing, staging, treatment pathways, risk stratification)
- "FIGO" — content restates FIGO staging criteria or FIGO guideline recommendations
- "ESGO" — content restates ESGO-ESTRO-ESP guideline recommendations
- "CSCO" — content restates CSCO guideline recommendations
- "ORIGINAL" — content is original analysis, commentary, or research findings from the paper itself, NOT restating any guideline
- "OTHER" — content comes from another identifiable source not listed above

Rules:
- If the content describes standard-of-care treatment pathways with specific regimens/doses, it's likely NCCN or ESGO
- If the content describes staging definitions, it's likely FIGO
- If the content presents original data analysis, statistics, or the paper's own conclusions, it's ORIGINAL
- Be specific: only attribute to a guideline if the content clearly restates guideline-level recommendations

Fragments to analyze:
{fragments_text}

Return ONLY a JSON object mapping fragment index (0-based) to the guideline name.
Example: {"0": "NCCN", "2": "ESGO", "4": "ORIGINAL"}
Do NOT include fragments where provenance is ORIGINAL — only return fragments that should be re-attributed to a specific guideline.
If none need re-attribution, return {{}}."""


class ResourceManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.moe_model: Optional[MoERouter] = None
        self.kmeans = None
        self.knn = None
        self.preprocessor = None
        self.patient_df = None
        self.xgb_classifiers = None
        self.X_vec: Optional[np.ndarray] = None
        self.patient_ids = None
        self.embed_client: Optional[AsyncOpenAI] = None
        self.rerank_client: Optional[AsyncOpenAI] = None
        self.llm_client: Optional[AsyncOpenAI] = None
        self.graph_engine: Optional[PathoRAG] = None
        self.figo_service: Optional[FigoService] = None
        self.embedding_func = None
        self._reranker_func = None

    async def initialize(self) -> None:
        os.environ["NEO4J_URI"] = self.config.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self.config.neo4j_username
        os.environ["NEO4J_PASSWORD"] = self.config.neo4j_password
        os.environ["NEO4J_DATABASE"] = self.config.neo4j_database
        os.environ["MILVUS_URI"] = self.config.milvus_uri

        init_attention_system(
            model_path=self.config.attention_model_path,
            vdb_path=self.config.vdb_entities_path,
            embedding_dim=self.config.embedding_dim,
        )

        self.embed_client = AsyncOpenAI(base_url=self.config.embedding_api_url, api_key=self.config.embedding_api_key)
        self.rerank_client = AsyncOpenAI(base_url=self.config.rerank_api_url, api_key=self.config.rerank_api_key)
        self.llm_client = AsyncOpenAI(base_url=self.config.llm_base_url, api_key=self.config.llm_api_key)

        @wrap_embedding_func_with_attrs(embedding_dim=self.config.embedding_dim, max_token_size=8192)
        async def embedding_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            response = await self.embed_client.embeddings.create(
                model=self.config.embedding_model_name, input=texts
            )
            return np.array([data.embedding for data in response.data])

        self.embedding_func = embedding_func

        async def vector_stream_reranker(query_str, docs_list):
            if not docs_list:
                return []
            scores = await asyncio.gather(*[
                compute_rerank_score(query_str, doc, self.rerank_client, self.config.rerank_model_name)
                for doc in docs_list
            ])
            return [
                doc
                for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True)
                if score > 0.01
            ]

        self._reranker_func = vector_stream_reranker

        self.graph_engine = PathoRAG(
            working_dir=self.config.working_dir,
            embedding_func=embedding_func,
            kv_storage="JsonKVStorage",
            vector_storage="MilvusVectorDBStorge",
            graph_storage="Neo4JStorage",
            reranker_func=vector_stream_reranker,
        )

        self.moe_model = MoERouter(input_dim=self.config.embedding_dim).to(DEVICE)
        checkpoint = torch.load(self.config.moe_model_path, weights_only=True)
        self.moe_model.load_state_dict(checkpoint, strict=False)
        self.moe_model.eval()

        loop = asyncio.get_running_loop()
        model_dir = self.config.v4_model_dir
        self.preprocessor = await loop.run_in_executor(
            ML_THREAD_POOL, joblib.load, str(model_dir / "preprocessor_retriever_v4.pkl")
        )
        self.kmeans = await loop.run_in_executor(
            ML_THREAD_POOL, joblib.load, str(model_dir / "kmeans_retriever_v4.pkl")
        )
        self.knn = await loop.run_in_executor(
            ML_THREAD_POOL, joblib.load, str(model_dir / "knn_index_retriever_v4.pkl")
        )
        self.patient_df = await loop.run_in_executor(
            ML_THREAD_POOL, pd.read_pickle, str(model_dir / "df_retriever_v4.pkl")
        )
        self.patient_ids = await loop.run_in_executor(
            ML_THREAD_POOL, joblib.load, str(model_dir / "patient_ids_retriever_v4.pkl")
        )
        self.X_vec = await loop.run_in_executor(
            ML_THREAD_POOL, np.load, str(model_dir / "X_vec_retriever_v4.npy")
        )
        xgb_path = model_dir / "trained_xgb_v4.pkl"
        if xgb_path.exists():
            self.xgb_classifiers = await loop.run_in_executor(
                ML_THREAD_POOL, joblib.load, str(xgb_path)
            )

        # ── FIGO 分期服务 ──────────────────────────────────────
        self.figo_service = FigoService(
            base_url=self.config.figo_vllm_base_url,
            model_name=self.config.figo_vllm_model_name,
            fallback_enabled=self.config.figo_fallback_enabled,
            fallback_api_key=self.config.llm_api_key,
            fallback_base_url=self.config.llm_base_url,
            fallback_model_name=self.config.llm_model_name,
        )
        self.figo_service.initialize()

    async def shutdown(self) -> None:
        if self.embed_client:
            await self.embed_client.close()
        if self.rerank_client:
            await self.rerank_client.close()
        if self.llm_client:
            await self.llm_client.close()
        ML_THREAD_POOL.shutdown(wait=True)
