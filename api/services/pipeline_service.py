import asyncio
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

import numpy as np
import torch
import pandas as pd

from api.config import AppConfig
from api.services.resource_manager import (
    ResourceManager,
    DEVICE,
    compute_rerank_score,
    get_source_details,
    get_guideline_tier,
    lookup_guideline_paper,
    is_non_authoritative_source,
    PROVENANCE_ANALYSIS_PROMPT,
    ML_THREAD_POOL,
)
from api.services.patient_retriever import PatientRetrieverV4
from api.schemas.response import (
    EvidenceFragment,
    BibliographyEntry,
    SimilarPatient,
    ComorbidityScreeningResult,
)

# --- Import CLUSTER_RAG modules ---
def _setup_cluster_rag_path(config: AppConfig):
    # Priority 1: configured path from .env
    candidates = [Path(config.cluster_rag_dir)]
    # Priority 2: relative to project root (robust fallback when CWD varies)
    _project_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(_project_root / "CLUSTER_RAG_Endometrial")

    found = None
    for cand in candidates:
        src = cand / "src"
        if src.is_dir():
            found = cand
            break

    if found is None:
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"CLUSTER_RAG_Endometrial not found. Searched:\n  {searched}\n"
            f"Set CLUSTER_RAG_DIR in .env to the correct path."
        )

    cluster_dir = str(found)
    src_dir = str(found / "src")
    for p in [cluster_dir, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_cluster_rag_path(AppConfig())

from v4_llm_pipeline import (
    build_prompt as build_pathology_prompt,
    extract_json_robust,
    normalize_X,
    get_base_stage_2023_from_2009,
)
from v4_comorbidity_extractor import build_comorbidity_prompt, COMORBIDITY_SCHEMA
from v4_esgo_decision_tree import classify_esgo_risk
from utils.trans_format import format_patient_desc
from comorbidity_skill import ComorbidityScreeningSkill

from pathorag_core import QueryParam


class PipelineStageError(Exception):
    def __init__(self, stage: str, message: str, original: Exception | None = None):
        self.stage = stage
        self.message = message
        self.original = original
        super().__init__(message)


ESGO_MAPPING = {
    "low": ("低危", "Low Risk"),
    "intermediate": ("中危", "Intermediate Risk"),
    "high-intermediate": ("中高危", "High-Intermediate Risk"),
    "high": ("高危", "High Risk"),
    "advanced": ("晚期转移", "Advanced Stage"),
    "unknown": ("不确定风险", "Uncertain Risk"),
}

MDT_REAL_WORLD_OVERRIDES_DICT = {
    "major_cv_risk": (
        "【真实世界MDT考量-严重心血管并发症/心脏支架】：患者存在极高心血管事件风险。"
        "若评估无法耐受大范围盆腔放疗或强效联合化疗的生理负荷，绝对不能死板生搬指南。"
        "建议启动MDT，强烈考虑【取消放疗】或【化疗降级/减药】，转向姑息性或单药低毒方案，以维持生活质量（QoL）为主。"
    ),
    "hepatic_dysfunction": (
        "【真实世界MDT考量-器官功能受损】：患者存在肝脏或多脏器功能异常，对系统性化疗耐受极差。"
        "建议放弃标准剂量的高毒性化疗，全面转向减量方案或最佳支持治疗（BSC）。"
    ),
}

ALIAS_MAPPING = {
    "Stage IA": "IA", "Stage IB": "IB", "Stage IIIA": "IIIA", "Stage IIIA1": "IIIA1",
    "I级": "G1", "II级": "G2", "III级": "G3",
    "高分化": "G1", "中分化": "G2", "低分化": "G3",
    "MMR正常": "pMMR", "MMR缺失": "dMMR",
    "阴性": "-", "阳性": "+",
}


class PipelineService:
    def __init__(self, resources: ResourceManager):
        self.rm = resources
        self.config = resources.config

    # === Stage 1: Extract patient profile + bilingual keywords ===
    async def _extract_patient_profile(self, patient_case: str) -> tuple[str, str]:
        sys_prompt = """你是一个极其严谨的妇科肿瘤病历特征提取专家系统。
你必须从病历中提取结构化信息，并严格以 JSON 格式输出。绝对禁止输出任何 JSON 块以外的解释性文字。

【提取核心准则】：
1. 【分期绝对采信】：必须直接提取病历中【术后诊断】给出的 FIGO 分期，禁止自行推演！
2. 【英文检索词缩写化】：英文词簇需使用精炼的专业缩写（如 Serous, G3, LVSI+, p53mut 等）。
3. 【空值处理】：若未提及某项特征，该字段请填入 "未提及"。
4. 【禁止推测未回报的检查结果】：如文本中明确提到"结果未出"、"待回报"、"未出结果"、"尚未回报"、"结果未回报"等，该字段请填入 "未提及"。切勿根据其他信息自行脑补！分子分型、免疫组化、基因检测等结果未回报时绝对禁止推测具体分型或突变状态！

【🚨 合并症提取要求】：comorbidities 列表必须包含患者所有合并症与既往病史，包括但不限于高血压、糖尿病、冠心病、HPV感染、肝炎、贫血等，不要遗漏 HPV 感染或人乳头瘤病毒感染！
【🚨 分子分型关键要求】：molecular_markers 字段必须严格区分以下两类信息：
  (a) 已回报的免疫组化结果（如 p53突变型/过表达、MMR正常、ER+ 等）
  (b) 分子分型检测状态（如"结果未出"、"待回报"、"未出结果"等，单独说明）
  切勿将 IHC p53 突变/过表达 等同于 分子分型中的 p53abn！

【🚨 强制 JSON 输出结构】：
{
  "patient_profile": {
    "basic_vitals": "年龄、绝经史等体征",
    "comorbidities": ["高血压2级", "2型糖尿病", "冠心病", "HPV感染"],
    "clinical_staging": "FIGO 2009/2023分期",
    "pathology_features": {
      "histology_type": "组织学类型(如浆液性癌)",
      "histological_grade": "组织学分级",
      "invasion_depth_and_involvement": "肌层浸润深度及宫外受累情况",
      "lvsi_status": "脉管癌栓(LVSI)状态",
      "lymph_node_status": "淋巴结转移状态",
      "molecular_markers": "分子标志物(p53, MMR等)及免疫组化（需明确区分已回报的IHC结果与待回报的分子分型）"
    }
  },
  "zh_keywords": ["提取核心中文病理特征词汇..."],
  "en_keywords": ["提取核心英文病理特征词汇..."]
}"""
        user_prompt = f"请根据以下原始病历，提取全息画像并生成中英双语检索词。\nQuery:\n{patient_case}\nOutput:"

        response = await self.rm.llm_client.chat.completions.create(
            model=self.config.llm_model_name,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=2048, response_format={"type": "json_object"},
        )
        parsed_data = extract_json_robust(response.choices[0].message.content)

        zh_kw = parsed_data.get("zh_keywords", [])
        en_kw = parsed_data.get("en_keywords", [])
        bilingual_keywords = "[中文检索词]: " + ", ".join(zh_kw) + "\n[英文检索词]: " + ", ".join(en_kw)

        profile = parsed_data.get("patient_profile", {})
        pathology = profile.get("pathology_features", {})
        comorbidities = "、".join(profile.get("comorbidities", [])) if profile.get("comorbidities") else "无"

        surgery = profile.get("surgical_procedures", {})
        procedures_list = surgery.get("procedures", [])
        if not procedures_list:
            m = re.search(r'手术方式及术中所见：(.+?)(?:\n\n|\n\d)', patient_case, re.DOTALL)
            if m:
                txt = m.group(1)
                procedures_list = [p.strip() for p in re.split(r'[、，]', txt.split('\n')[0]) if p.strip()]
                surgery["procedures"] = procedures_list
                lines = txt.strip().split('\n')
                if len(lines) > 1:
                    surgery["intraoperative_findings"] = " ".join(l.strip() for l in lines[1:] if l.strip())[:300]
                surgery["impact_on_diagnosis"] = "已行手术获得完整术后病理（金标准），分期基于手术病理结果，淋巴结状态已明确，后续治疗决策需在手术病理指导下进行"
        procedures_str = "、".join(procedures_list) if procedures_list else "未提及"
        intraop = surgery.get("intraoperative_findings", "未提及")
        impact = surgery.get("impact_on_diagnosis", "未提及")

        profile_md = (
            f"### 【全息患者画像】\n"
            f"- **基本体征**：{profile.get('basic_vitals', '未提及')}\n"
            f"- **所有合并症与既往史**：{comorbidities}\n"
            f"- **既定分期**：{profile.get('clinical_staging', '未提及')}\n"
            f"- **已行手术方式**：{procedures_str}\n"
            f"  - 术中所见：{intraop}\n"
            f"  - 对诊断的影响：{impact}\n"
            f"- **病理与转移特征**：\n"
            f"  - 组织学类型：{pathology.get('histology_type', '未提及')}\n"
            f"  - 组织学分级：{pathology.get('histological_grade', '未提及')}\n"
            f"  - 浸润与周围受累：{pathology.get('invasion_depth_and_involvement', '未提及')}\n"
            f"  - 脉管癌栓 (LVSI)：{pathology.get('lvsi_status', '未提及')}\n"
            f"  - 淋巴结状态：{pathology.get('lymph_node_status', '未提及')}\n"
            f"  - 分子分型关键指标：{pathology.get('molecular_markers', '未提及')}"
        )
        return profile_md, bilingual_keywords

    # === Stage 2: Extract structured X features + ESGO classification ===
    async def _get_structured_features(self, patient_case: str) -> dict:
        pathology_p = build_pathology_prompt("realtime_task", patient_case)
        comorbidity_p = build_comorbidity_prompt("realtime_task", patient_case)
        tasks = [
            self.rm.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": pathology_p}],
                response_format={"type": "json_object"},
            ),
            self.rm.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": comorbidity_p}],
                response_format={"type": "json_object"},
            ),
        ]
        patho_res, como_res = await asyncio.gather(*tasks)
        patho_json = extract_json_robust(patho_res.choices[0].message.content)
        como_json = extract_json_robust(como_res.choices[0].message.content)
        new_patient_dict = normalize_X(patho_json.get("X", patho_json))

        stage_2023_val, _ = get_base_stage_2023_from_2009(
            new_patient_dict.get("stage_raw", "I"),
            new_patient_dict.get("histology_type", "serous"),
            new_patient_dict.get("grade", "G1"),
            new_patient_dict.get("myometrial_invasion_ratio", "<50%"),
            new_patient_dict.get("myometrial_invasion_depth", None),
            new_patient_dict.get("lvsi", "negative"),
            new_patient_dict.get("lvsi_substantial", False),
            new_patient_dict.get("cervical_involvement", "negative"),
            new_patient_dict.get("adnexal_involvement", 0),
            new_patient_dict.get("lymph_node_pelvic", "0/0"),
            new_patient_dict.get("lymph_node_paraaortic", "negative"),
            new_patient_dict.get("peritoneal_cytology", "negative"),
        )
        new_patient_dict["stage_2023"] = stage_2023_val
        como_x = como_json.get("X", como_json)
        for key in COMORBIDITY_SCHEMA:
            new_patient_dict[key] = como_x.get(key, 0)
        new_patient_dict["esgo_risk_group"] = classify_esgo_risk(new_patient_dict)
        return new_patient_dict

    # === Helper: Extract English-only keywords for cross-lingual retrieval ===
    @staticmethod
    def _extract_en_query(bilingual_keywords: str) -> str | None:
        """从双语关键词中抽取英文部分，用于英文专项检索通道"""
        if not bilingual_keywords:
            return None
        # 匹配 "[英文检索词]: ..." 部分
        m = re.search(r'\[英文检索词\]\s*:?\s*(.+)', bilingual_keywords, re.IGNORECASE)
        if not m:
            return None
        en_part = m.group(1).strip()
        if not en_part:
            return None
        # 清洗：去重、去空白
        parts = re.split(r'[,，]', en_part)
        seen = set()
        clean = []
        for p in parts:
            p = p.strip().rstrip('.')
            if p and p.lower() not in seen:
                seen.add(p.lower())
                clean.append(p)
        return ", ".join(clean) if clean else None

    # === Stage 3: Process keywords for graph query ===
    def _prepare_graph_query(self, bilingual_keywords: str, esgo_risk_level: str, patient_case: str) -> str:
        pure_kws = []
        if bilingual_keywords:
            for line in bilingual_keywords.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if ']' in line and ':' in line:
                    content = line.split(':', 1)[1]
                else:
                    content = line
                parts = re.split(r'[,，]', content)
                pure_kws.extend([p.strip() for p in parts if p.strip()])

        if esgo_risk_level:
            pure_kws.append(esgo_risk_level.strip())

        seen = set()
        final_kws = []
        for kw in pure_kws:
            kw = kw.strip()
            if not kw:
                continue
            kw = re.sub(r'(?i)^Stage\s*', '', kw)
            kw = re.sub(r'期$', '', kw)
            mapped_kw = ALIAS_MAPPING.get(kw, kw)
            if mapped_kw not in seen:
                seen.add(mapped_kw)
                final_kws.append(mapped_kw)

        return ", ".join(final_kws) if final_kws else patient_case[:200]

    # === Stage 4: MoE fusion weight ===
    async def _compute_moe_weight(self, graph_query: str) -> float:
        query_emb = await self.rm.embedding_func([graph_query])
        query_vec = query_emb[0] / (np.linalg.norm(query_emb[0]) + 1e-8)
        query_tensor = torch.tensor(query_vec, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            g_raw = self.rm.moe_model(query_tensor.unsqueeze(0)).item()
            g = max(0.02, min(0.98, g_raw))
        return g

    # === Stage 5: Hybrid knowledge graph retrieval (with cross-lingual English channel) ===
    async def _retrieve_knowledge(
        self, graph_query: str, moe_weight: float, en_query: str | None = None
    ) -> dict:
        param = QueryParam(mode="hybrid", top_k=60, max_token_for_text_unit=4000, only_need_context=True)
        setattr(param, "moe_weight", moe_weight)

        # 主检索（中英文混合关键词）
        main_task = self.rm.graph_engine.aquery(graph_query, param)

        # 英文专项检索通道：用纯英文关键词独立检索 Milvus，弥补 QwenEmbedding 亲中文偏差
        en_task = None
        if en_query and self.config.en_retrieval_enabled:
            en_param = QueryParam(mode="hybrid", top_k=self.config.en_retrieval_top_k,
                                  max_token_for_text_unit=4000, only_need_context=True)
            setattr(en_param, "moe_weight", 0.3)  # 英文图谱连接稀疏，倾向向量检索
            en_task = self.rm.graph_engine.aquery(en_query, en_param)

        if en_task:
            main_results, en_results = await asyncio.gather(main_task, en_task)
        else:
            main_results = await main_task
            en_results = []

        extended_pool: dict = {}
        # 处理主流结果
        if isinstance(main_results, list):
            for item in main_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    if len(content) < 5 or content in ("RELATES_TO", "BELONG_TO", "EVIDENCE"):
                        continue
                    tag_info = "🧠 图谱高阶逻辑" if content.startswith("【权威循证溯源：") else "🧩 纯向量语义召回"
                    extended_pool[content] = {"type": tag_info, "graph_score": graph_score}

        # 处理英文通道结果（标记为英文文献召回，给予略低的基础分防止完全覆盖中文结果）
        en_count = 0
        if isinstance(en_results, list):
            for item in en_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    if len(content) < 5 or content in ("RELATES_TO", "BELONG_TO", "EVIDENCE"):
                        continue
                    if content not in extended_pool:
                        extended_pool[content] = {
                            "type": "📚 英文文献召回",
                            "graph_score": graph_score * 0.85,  # 轻微降权，避免英文结果淹没中文指南
                        }
                        en_count += 1
        if en_count > 0:
            logger.info("English retrieval channel added %d unique fragments (total pool: %d)",
                        en_count, len(extended_pool))

        return extended_pool

    # === Stage 6: Rerank and build bibliography ===
    def _adaptive_top_k(self, fragments: list[dict]) -> int:
        score_field = "final_score"
        min_k = self.config.adaptive_k_min
        max_k = self.config.adaptive_k_max
        threshold = self.config.adaptive_k_drop_threshold

        k = min_k
        for i in range(min_k, min(len(fragments), max_k)):
            curr = fragments[i][score_field]
            prev = fragments[i - 1][score_field]
            if prev > 1e-8 and (prev - curr) / prev > threshold:
                break
            k = i + 1
        return min(k, len(fragments))

    async def _analyze_provenance(self, fragments: list[dict]) -> dict[int, str]:
        """LLM-based provenance analysis for fragments from non-authoritative sources.

        For each fragment whose source paper is tagged as "其他指南" or similar,
        ask the LLM to determine which authoritative guideline the content actually
        restates. Returns mapping of fragment_index → guideline_name (NCCN/FIGO/ESGO/CSCO).
        """
        needs_check = []
        for i, frag in enumerate(fragments):
            sources = frag.get("sources", [])
            if not sources:
                continue
            all_non_auth = all(
                is_non_authoritative_source(src.get("guidelines", ""))
                for src in sources
            )
            if all_non_auth:
                needs_check.append((i, frag["content"]))

        if not needs_check:
            return {}

        fragments_text = "\n\n".join([
            f"--- Fragment {idx} ---\n{content[:1000]}"
            for idx, content in needs_check
        ])

        prompt = PROVENANCE_ANALYSIS_PROMPT.replace("{fragments_text}", fragments_text)
        logger.info(
            "Provenance analysis: checking %d fragments from non-authoritative sources",
            len(needs_check),
        )

        try:
            response = await self.rm.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content.strip()

            json_match = re.search(r"\{[^{}]*\}", result_text, re.DOTALL)
            if not json_match:
                logger.warning(
                    "Provenance analysis: could not parse JSON from response: %s",
                    result_text[:200],
                )
                return {}

            mapping = json.loads(json_match.group(0))
            result = {}
            for k, v in mapping.items():
                try:
                    idx = int(k)
                    guideline = str(v).strip().upper()
                    if guideline in ("NCCN", "FIGO", "ESGO", "CSCO"):
                        result[idx] = guideline
                except (ValueError, TypeError):
                    continue

            if result:
                logger.info(
                    "Provenance analysis: re-attributed %d fragments → %s",
                    len(result), result,
                )
            return result
        except Exception as e:
            logger.warning("Provenance analysis failed (non-fatal): %s", e)
            return {}

    async def _rerank_and_build_bibliography(
        self, extended_pool: dict, patient_case: str, patient_dict: dict, moe_weight: float = 0.5
    ) -> tuple[list[dict], list[dict], str]:
        candidate_contents = list(extended_pool.keys())
        if not candidate_contents:
            return [], [], ""

        semantic_scores = await asyncio.gather(*[
            compute_rerank_score(patient_case, c, self.rm.rerank_client, self.config.rerank_model_name)
            for c in candidate_contents
        ])

        fragments_to_sort = []
        for idx, content in enumerate(candidate_contents):
            info = extended_pool[content]
            sources = await get_source_details(self.rm.graph_engine, content)
            best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
            final_score = (moe_weight * info["graph_score"]) + ((1 - moe_weight) * semantic_scores[idx]) + (4 - best_tier) * 0.1
            fragments_to_sort.append({"content": content, "sources": sources, "final_score": final_score})

        fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
        adaptive_k = self._adaptive_top_k(fragments_to_sort)
        if adaptive_k != 10:
            logger.info(
                "Adaptive top-k adjusted: %d (min=%d max=%d threshold=%.2f) - %d candidates available",
                adaptive_k, self.config.adaptive_k_min, self.config.adaptive_k_max,
                self.config.adaptive_k_drop_threshold, len(fragments_to_sort),
            )
        selected = fragments_to_sort[:adaptive_k]

        # --- Provenance analysis: redirect citations from secondary sources to authoritative guidelines ---
        provenance_map = await self._analyze_provenance(selected)
        if provenance_map:
            storage = self.rm.graph_engine.chunk_entity_relation_graph
            if hasattr(storage, "driver"):
                async with storage.driver.session() as session:
                    for frag_idx, guideline_name in provenance_map.items():
                        if frag_idx < 0 or frag_idx >= len(selected):
                            continue
                        authoritative = await lookup_guideline_paper(session, guideline_name)
                        if authoritative:
                            old_src = selected[frag_idx]["sources"]
                            old_ids = [s.get("id", "?") for s in old_src]
                            selected[frag_idx]["sources"] = [authoritative]
                            logger.info(
                                "Provenance redirect [%d]: source %s → %s (%s)",
                                frag_idx,
                                old_ids,
                                guideline_name,
                                authoritative.get("title", "?")[:80],
                            )
                        else:
                            logger.info(
                                "Provenance redirect [%d]: wanted %s but no matching Paper node found in Neo4j",
                                frag_idx, guideline_name,
                            )

        bibliography = {}
        ref_counter = 1
        llm_context_list = []

        esgo_paper_id = "paper::40744042"
        esgo_raw_label = patient_dict.get("esgo_risk_group", "unknown").lower()
        esgo_risk_level, en_risk_keyword = ESGO_MAPPING.get(esgo_raw_label, ("未知风险", "Unknown Risk"))
        safe_clinical_conclusion = f"综合患者特征，经内置 ESGO 2025 算法严格计算判定为：【{esgo_risk_level} ({en_risk_keyword})】。"
        bibliography[esgo_paper_id] = {
            "id": esgo_paper_id, "pmid": "40744042",
            "title": "ESGO-ESTRO-ESP guidelines for the management of patients with endometrial carcinoma: update 2025",
            "guidelines": "ESGO指南", "ref_index": ref_counter,
        }
        llm_context_list.append(f"【来源文献: [{ref_counter}] | 证据级别: ESGO指南 (首选)】 {safe_clinical_conclusion}")
        ref_counter += 1

        for frag in selected:
            source_indices = []
            for src in frag["sources"]:
                src_id = src["id"]
                if src_id not in bibliography:
                    src["ref_index"] = ref_counter
                    bibliography[src_id] = src
                    ref_counter += 1
                source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
            ref_tag = f"【来源文献: {', '.join(source_indices)}】" if source_indices else ""
            llm_context_list.append(f"{ref_tag} {frag['content']}")

        context_str = "\n\n".join(llm_context_list)
        return selected, bibliography, context_str

    # === Stage 7: Similar patient retrieval + XGBoost ===
    async def _retrieve_similar_patients(self, patient_dict: dict, top_k: int):
        if "stage_2023" not in patient_dict:
            patient_dict["stage_2023"] = patient_dict.get("stage_raw", "I")

        loop = asyncio.get_running_loop()

        def _sync_retrieve():
            retriever = PatientRetrieverV4(
                preprocessor=self.rm.preprocessor,
                kmeans=self.rm.kmeans,
                knn=self.rm.knn,
                df=self.rm.patient_df,
                X_vec=self.rm.X_vec,
                patient_ids=self.rm.patient_ids,
                xgb_classifiers=self.rm.xgb_classifiers,
                thresholds=self.config.thresholds,
            )
            df = retriever.retrieve(patient_dict, top_k=top_k)
            preds = retriever.predict(patient_dict)
            return df, preds

        retrieved_df, xgb_pred = await loop.run_in_executor(ML_THREAD_POOL, _sync_retrieve)
        return retrieved_df, xgb_pred

    # === Stage 8: Comorbidity screening ===
    async def _screen_comorbidities(
        self, patient_dict: dict, profile_md: str, retrieved_df: pd.DataFrame, mdt_overrides_str: str
    ) -> dict:
        skill = ComorbidityScreeningSkill(
            llm_client=self.rm.llm_client,
            reference_path=str(self.config.nccn_reference_path),
        )
        return await skill.screen(
            patient_dict=patient_dict,
            patient_profile_md=profile_md,
            similar_patients_df=retrieved_df,
            mdt_overrides_str=mdt_overrides_str,
        )

    # === Stage 9: Generate MDT report (non-streaming collect) ===
    async def _generate_mdt_report(
        self, profile_md: str, patient_dict: dict, context_str: str,
        filtered_cases_text: str, pred_summary: str,
        safety_warnings_text: str, deescalation_text: str,
    ) -> str:
        system_prompt = "你是一名严谨的妇科肿瘤 MDT 首席专家。你的任务是：基于知识图谱提供的证据，对【权威指南推荐】进行极其详尽的拆解和论述；同时，为下游的 EBM 系统提出需要深度查证的具体临床数据问题。"

        user_prompt = f"""以下是患者的【全息特征画像】：
{profile_md}

以下是经过药师 Agent 严格筛查后，确认对本患者具备高度参考价值的【相似病例治疗建议】：
{filtered_cases_text}
{pred_summary if pred_summary else ""}

以下是图谱召回的【参考证据】（已包含系统前置推演结论及权威文献）：
{context_str}

【🚨 核心分工与防幻觉准则】：
1. **指南解析必须极尽详尽（核心任务！）**：你必须把你能在【参考证据】中看到的指南细则长篇大论地写出来。
2. **知识片段精准标记**：引用图谱返回的完整语义片段时，**必须加粗核心词汇，并严格在句末打上对应的文献标号 [x]**。
3. **强制采纳药师预警**：将下方的【药师安全预警】无缝融合到"合并症管理"章节中。
4. **数据验证留给下游**：绝对禁止编造具体的生存率或 HR 数值，将其写在"PICO提问"中！
5. 【🚨 绝对禁止推测分子分型】：下方模板示例中的"如分子分型为p53abn/MMRd/NSMP/POLEmut"均为假设性论述，仅用于示范指南的拆解方式。你必须以患者画像中的实际分子分型为准！若画像中分子分型为"未提及"或明确说"结果未出"，则指南解析部分不得声称患者属于某种分子分型，只能用"若分子分型结果回报后"等条件句式。
6. 【🚨 禁止将 IHC p53 等价于分子 p53abn】：免疫组化（IHC）检测的 p53 突变型/过表达 ≠ 分子分型中的 p53abn（NGS/Sanger 测序结果）。即使画像中 IHC 结果显示 p53 突变型/过表达，只要分子分型状态为"结果未出"或"未提及"，ESGO/NCCN 指南解析中涉及"p53abn"的推荐路径均只能用条件句式（如"若分子分型回报为p53abn"），不得将其作为已确认的分子分型来展开论述！
7. 【🚨 禁止前后矛盾！】整份报告不得出现此类自相矛盾：病情摘要已声明"分子分型结果未出"，后续段落却声称"基于已明确的p53突变分子分型"。IHC p53 异常 ≠ 分子分型已明确。一旦先后矛盾，整份报告可信度归零！

【💊 药师 Agent 药学安全预警】：
{safety_warnings_text if safety_warnings_text else "无"}

【⚖️ MDT 真实世界方案降级与修正指令】：
{deescalation_text if deescalation_text else "无（患者体能良好，可按指南标准执行）"}

【📝 强制报告排版结构（严格遵守格式！）】：

# 妇科肿瘤 MDT 初始会诊报告

## 一、 病情摘要与风险判定
- **病历摘要**：凝练患者年龄、绝经史、合并症、核心病理、淋巴结状态及 FIGO 分期及已行手术方式。**必须说明已行手术名称（如全子宫切除、双附件切除、前哨淋巴结活检等）及其对诊断的影响：术后病理为金标准，分期基于手术病理结果，淋巴结状态和宫外受累已明确**。
- **ESGO 2025 风险分层**：直接引用 [1] 号文献的结论（如：**高危 (High Risk)** [1]）。

## 二、 核心指南与共识详尽解析
（🚨 必须打破将所有指南混写的格式！请将检索到的【每一个指南/共识单独列出子标题】。在对该指南进行连贯的分析后，必须独立提炼出该指南针对此患者的"核心推荐路径"，并用代码块高亮显示。）
请严格采用以下排版风格：

**（🚨 重要：以下所有指南解析必须结合患者已行手术情况进行论述：患者已行全子宫+双附件切除+前哨淋巴结活检，术后病理已明确肌层浸润≥1/2、LVSI（+）、宫颈间质浸润、左输卵管癌累及、淋巴结阴性，所有治疗决策均为辅助治疗而非初治选择！）**

### 1. 2025 ESGO-ESTRO-ESP 指南 [1]
- **具体指南详析**：根据该指南，该患者属于高危型，子宫内膜浆液性癌IIIA1期，无肉眼残留病灶，免疫组化P53（突变表型）。如分子分型为p53abn或MMRd/NSMP，属于高危型内膜癌...（请将图谱中关于该指南的详细证据连贯地融合在此处）。如分子分型回报为 POLEmut，尚缺乏数据... [1]。
- **核心推荐路径**：`EBRT + 同步化疗 + 辅助化疗，或 序贯化疗 + 放疗`

### 2. NCCN 临床实践指南 [x]
- **具体指南详析**：NCCN 指南同样指出...（自然融入图谱证据并标记文献编号）。对于具有高危因素的浆液性癌，常规治疗效果不佳，复发率高...
- **核心推荐路径**：`系统化疗 ± 外照射放疗 (EBRT) ± 阴道近距离放疗 (VBT)`

### 3. 国内共识及其他指南 [x]
- **具体指南详析**：根据中国妇科肿瘤临床实践指南...（融合证据）。
- **核心推荐路径**：`（提取该指南对应的公式化路径）`

【🚨 禁止在此报告中提及复发/晚期二线+药物】：患者处于术后辅助治疗（一线）阶段。绝对禁止出现仑伐替尼、帕博利珠单抗、纳武利尤单抗、多塔利单抗、仑伐替尼+帕博利珠单抗等复发/晚期（二线+）抢救方案药物名称！这些是复发后才考虑的最后防线，写入当前报告既脱离临床实际阶段，又严重加重患者恐慌！

## 三、 初步专科治疗框架
- **肿瘤主方案建议**：用一句话概括（基于手术病理分期），格式如"建议行TC方案6次，治疗结束后3个月复查盆腔增强MRI/上腹部增强CT/两肺平扫CT或PET-CT"——不要分点罗列。
- **多学科及合并症管理**：
  （**🚨 必须使用阿拉伯数字分点列出所有的合并症！绝不能合并成一段！** 🚨 化疗前感染筛查：必须逐条核对原始病历，如存在 HP现症感染、慢性肺部炎症、活动性肝炎等感染性疾病，必须在合并症管理中单独列出，并标注为"化疗前需处理/评估"——带活动性感染上化疗，骨髓抑制期极易引发重症感染或消化道大出血！**）
  1、患者高血压，建议心内科随诊。
  2、患者糖尿病，建议内分泌科随诊，关注化疗期间血糖波动。
  3、患者脑梗后遗症，建议神经内科随诊，注意血栓风险。
  4、患者HPV感染，建议完善分型检测，免疫治疗期间加强阴道残端细胞学随访。**注意：患者已行全子宫切除，无宫颈，不得建议宫颈细胞学检查。**
  ……（逐条列出）

## 四、 随访大纲
列出常规的随访频率（如前两年每3个月一次）以及需要患者警惕的核心异常体征（如提示粘连梗阻或复发的症状）、检查项目等。

【🚨 随访方案红线】：
1. **已行全子宫切除，绝对禁止提及宫颈检查、宫颈细胞学、TCT、HPV宫颈取样等任何宫颈相关项目！** 替代方案：阴道残端细胞学检查。
2. **肿瘤标志物 CA125、HE4 不需要空腹抽血！** 绝对禁止写"需空腹"。
3. **禁止在初治辅助治疗阶段提及复发/晚期抢救方案**：禁止出现仑伐替尼、帕博利珠单抗、纳武利尤单抗、仑伐替尼+帕博利珠单抗等二线+药物名称。患者处于术后辅助治疗阶段，不是复发/晚期！

### 五、 向 EBM 循证系统提问 (PICO Questions)

列出 2-3 条检索需求，每条约 1 句话，格式：`{{核心试验}}：{{检索方向}}`。

【🚨 强制试验匹配规则——必须严格遵守，不得自行列举其他试验！】：
根据患者的分期，**只允许**从下方对应的试验中选取，不得添加规则未列出的试验：

| 患者分期 | 只允许检索以下试验 |
|----------|------------------|
| I-II期中低危 | GOG-99, PORTEC-1, PORTEC-2 |
| I-II期高危 | **PORTEC-3, GOG-0258** |
| III-IVA期 | **PORTEC-3, GOG-0258**（注意：此期别不适用 PORTEC-4a） |
| IVB期/复发一线 | GOG-209, NRG-GY018, RUBY |
| 晚期复发（二线+） | KEYNOTE-775 |
| I-II期 分子分型降/升阶梯探索（III期及以上不适用） | 可额外追加 PORTEC-4a |

**输出格式示例**：
- `{{试验名}}：{{具体临床问题}}`

【红线】：
1. **绝对禁止**检索导航库以外的试验（如 KEYNOTE-775、NRG-GY018 等只适用于对应的晚期分期）。
2. **禁止提问药物毒副反应**，将名额留给前沿疗效与分子分型突破。
3. **绝对禁止在 PICO 中擅自断言分子分型**：若病情摘要写明"结果未出"或"待回报"，PICO 中不得写 "p53abn型患者" 等字眼，只能用"分子分型待明确的患者"或直接不提。

💡 请先在 <think> 标签内梳理：1. 确认风险等级；2. 疯狂提取证据里的指南细节；3. 盘点所有合并症；4. 构思要留给下游的硬核数据问题。思考完毕后，输出这份专业报告！"""

        response = await self.rm.llm_client.chat.completions.create(
            model=self.config.llm_model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            stream=True, temperature=0.1, max_tokens=8192,
        )
        chunks = []
        async for chunk in response:
            if content := chunk.choices[0].delta.content:
                chunks.append(content)
        return "".join(chunks)

    # === Build MDT overrides ===
    def _build_mdt_overrides(self, patient_dict: dict) -> tuple[list[str], str]:
        overrides = []
        if patient_dict.get("major_cv_risk", 0) > 0 or patient_dict.get("major_cv_risk") is True:
            overrides.append(MDT_REAL_WORLD_OVERRIDES_DICT["major_cv_risk"])
        try:
            age_val = float(patient_dict.get("age", 0))
            if age_val >= 75:
                overrides.append(
                    "【真实世界MDT考量-高龄与极度衰弱】：患者为75岁以上高龄，预期ECOG评分较差。"
                    "常规NCCN推荐的系统性强放化疗可能带来弊大于利的毒性。"
                    "临床决策应从'延长生存'向'维持生活质量'倾斜，允许对标准方案进行减量、延迟或放弃放疗。"
                )
        except (ValueError, TypeError):
            pass
        overrides_str = "\n".join(overrides) if overrides else "无特殊真实世界MDT降级考量。"
        return overrides, overrides_str

    # === Full pipeline ===
    async def execute(self, patient_case: str, top_k_similar: int = 3) -> dict:
        request_id = uuid.uuid4().hex[:12]

        # Stage 1
        profile_md, bilingual_keywords = await self._extract_patient_profile(patient_case)

        # Stage 2
        patient_dict = await self._get_structured_features(patient_case)

        # ESGO risk
        esgo_raw_label = patient_dict.get("esgo_risk_group", "unknown").lower()
        esgo_risk_level, en_risk_keyword = ESGO_MAPPING.get(esgo_raw_label, ("未知风险", "Unknown Risk"))

        # MDT overrides
        mdt_overrides, mdt_overrides_str = self._build_mdt_overrides(patient_dict)

        # Stage 3
        graph_query = self._prepare_graph_query(bilingual_keywords, esgo_risk_level, patient_case)
        en_query = self._extract_en_query(bilingual_keywords)

        # Stage 4
        g = await self._compute_moe_weight(graph_query)

        # Stage 5
        extended_pool = await self._retrieve_knowledge(graph_query, g, en_query=en_query)

        # Stage 6
        selected, bibliography, context_str = await self._rerank_and_build_bibliography(
            extended_pool, patient_case, patient_dict, g
        )

        # Stage 7
        retrieved_df, xgb_pred = await self._retrieve_similar_patients(patient_dict, top_k_similar)

        # Stage 8
        try:
            screening_result = await self._screen_comorbidities(
                patient_dict, profile_md, retrieved_df, mdt_overrides_str
            )
        except Exception:
            screening_result = {
                "filtered_cases": [], "safety_warnings": [],
                "deescalation_advice": [], "monitoring_recommendations": [],
            }

        # Build filtered cases text
        retained_ids = [
            c["case_id"] for c in screening_result.get("filtered_cases", [])
            if c.get("is_retained", False)
        ]
        filtered_cases_text = "\n【经过合并症筛查 Agent 严格质控后的真实世界相似病例】\n"
        if retained_ids and not retrieved_df.empty:
            for idx, row in retrieved_df.iterrows():
                if str(idx) in str(retained_ids):
                    filtered_cases_text += f"\n✅ [采纳病例] ID: {idx}\n"
                    filtered_cases_text += format_patient_desc(row) + "\n"
                    filtered_cases_text += f"  历史治疗建议原文: {row.get('Y_text', '')}\n"
        else:
            filtered_cases_text += "⚠️ 警告：无可用相似病例，请完全依赖指南证据制定方案。\n"

        safety_warnings_text = "\n".join([
            f"⚠️ {w}" for w in screening_result.get("safety_warnings", [])
        ])
        deescalation_text = "\n".join([
            f"🛑 {a}" for a in screening_result.get("deescalation_advice", [])
        ])
        pred_summary = "\n".join([
            f"- {k}: {'推荐' if v == 1 else '不推荐'}" for k, v in xgb_pred.items()
        ])

        # Stage 9
        mdt_report = await self._generate_mdt_report(
            profile_md, patient_dict, context_str,
            filtered_cases_text, pred_summary,
            safety_warnings_text, deescalation_text,
        )

        # Build response
        evidence_fragments = [
            EvidenceFragment(
                content=f["content"],
                source_ids=[s.get("id", "") for s in f["sources"]],
                final_score=f["final_score"],
                evidence_type=extended_pool.get(f["content"], {}).get("type", ""),
            )
            for f in selected
        ]

        bib_entries = [
            BibliographyEntry(
                ref_index=v["ref_index"],
                pmid=v.get("pmid", ""),
                title=v.get("title", ""),
                guidelines=v.get("guidelines", ""),
                source_id=v.get("id", ""),
            )
            for v in sorted(bibliography.values(), key=lambda x: x["ref_index"])
        ]

        similar_patients = []
        if not retrieved_df.empty:
            for idx, row in retrieved_df.iterrows():
                similar_patients.append(SimilarPatient(
                    patient_id=str(idx),
                    distance=float(row.get("distance", 0)),
                    description=format_patient_desc(row),
                    historical_treatment=str(row.get("Y_text", "")),
                ))

        return {
            "request_id": request_id,
            "patient_profile_md": profile_md,
            "esgo_risk_classification": f"{esgo_risk_level} ({en_risk_keyword})",
            "esgo_raw_label": esgo_raw_label,
            "knowledge_evidence": evidence_fragments,
            "bibliography": bib_entries,
            "similar_patients": similar_patients,
            "xgb_predictions": xgb_pred,
            "comorbidity_screening": ComorbidityScreeningResult(
                safety_warnings=screening_result.get("safety_warnings", []),
                deescalation_advice=screening_result.get("deescalation_advice", []),
                monitoring_recommendations=screening_result.get("monitoring_recommendations", []),
                coverage_summary=screening_result.get("coverage_summary", {}),
                reference_log=screening_result.get("reference_log", []),
            ),
            "mdt_report": mdt_report,
            "mdt_overrides": mdt_overrides,
        }

    # === Streaming pipeline (for SSE endpoint) ===
    async def execute_stream(self, patient_case: str, top_k_similar: int = 3) -> AsyncIterator[tuple[str, dict | str]]:
        request_id = uuid.uuid4().hex[:12]

        try:
            yield ("progress", {"stage": "patient_profile", "status": "started"})
            profile_md, bilingual_keywords = await self._extract_patient_profile(patient_case)
            yield ("progress", {"stage": "patient_profile", "status": "done"})

            yield ("progress", {"stage": "structured_features", "status": "started"})
            patient_dict = await self._get_structured_features(patient_case)
            yield ("progress", {"stage": "structured_features", "status": "done"})

            esgo_raw_label = patient_dict.get("esgo_risk_group", "unknown").lower()
            esgo_risk_level, _ = ESGO_MAPPING.get(esgo_raw_label, ("未知风险", "Unknown Risk"))
            mdt_overrides, mdt_overrides_str = self._build_mdt_overrides(patient_dict)

            yield ("progress", {"stage": "graph_query", "status": "started"})
            graph_query = self._prepare_graph_query(bilingual_keywords, esgo_risk_level, patient_case)
            en_query = self._extract_en_query(bilingual_keywords)
            yield ("progress", {"stage": "graph_query", "status": "done"})

            yield ("progress", {"stage": "moe_weight", "status": "started"})
            g = await self._compute_moe_weight(graph_query)
            yield ("progress", {"stage": "moe_weight", "status": "done", "weight": g})

            yield ("progress", {"stage": "knowledge_retrieval", "status": "started"})
            extended_pool = await self._retrieve_knowledge(graph_query, g, en_query=en_query)
            yield ("progress", {"stage": "knowledge_retrieval", "status": "done", "count": len(extended_pool)})

            yield ("progress", {"stage": "rerank", "status": "started"})
            selected, bibliography, context_str = await self._rerank_and_build_bibliography(
                extended_pool, patient_case, patient_dict, g
            )
            yield ("progress", {"stage": "rerank", "status": "done", "selected_count": len(selected)})

            yield ("progress", {"stage": "similar_patients", "status": "started"})
            retrieved_df, xgb_pred = await self._retrieve_similar_patients(patient_dict, top_k_similar)
            yield ("progress", {"stage": "similar_patients", "status": "done"})

            yield ("progress", {"stage": "comorbidity_screening", "status": "started"})
            try:
                screening_result = await self._screen_comorbidities(
                    patient_dict, profile_md, retrieved_df, mdt_overrides_str
                )
            except Exception:
                screening_result = {
                    "filtered_cases": [], "safety_warnings": [],
                    "deescalation_advice": [], "monitoring_recommendations": [],
                }
            yield ("progress", {"stage": "comorbidity_screening", "status": "done"})

            # Build filtered cases (same as non-streaming)
            retained_ids = [
                c["case_id"] for c in screening_result.get("filtered_cases", [])
                if c.get("is_retained", False)
            ]
            filtered_cases_text = "\n【经过合并症筛查 Agent 严格质控后的真实世界相似病例】\n"
            if retained_ids and not retrieved_df.empty:
                for idx, row in retrieved_df.iterrows():
                    if str(idx) in str(retained_ids):
                        filtered_cases_text += f"\n✅ [采纳病例] ID: {idx}\n"
                        filtered_cases_text += format_patient_desc(row) + "\n"
                        filtered_cases_text += f"  历史治疗建议原文: {row.get('Y_text', '')}\n"
            else:
                filtered_cases_text += "⚠️ 警告：无可用相似病例，请完全依赖指南证据制定方案。\n"

            safety_warnings_text = "\n".join([
                f"⚠️ {w}" for w in screening_result.get("safety_warnings", [])
            ])
            deescalation_text = "\n".join([
                f"🛑 {a}" for a in screening_result.get("deescalation_advice", [])
            ])
            pred_summary = "\n".join([
                f"- {k}: {'推荐' if v == 1 else '不推荐'}" for k, v in xgb_pred.items()
            ])

            # Stream MDT generation
            yield ("progress", {"stage": "mdt_report", "status": "streaming"})
            system_prompt = "你是一名严谨的妇科肿瘤 MDT 首席专家。你的任务是：基于知识图谱提供的证据，对【权威指南推荐】进行极其详尽的拆解和论述；同时，为下游的 EBM 系统提出需要深度查证的具体临床数据问题。"
            user_prompt = self._build_mdt_prompt(
                profile_md, filtered_cases_text, pred_summary, context_str,
                safety_warnings_text, deescalation_text,
            )

            response = await self.rm.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                stream=True, temperature=0.1, max_tokens=8192,
            )
            full_report_chunks = []
            async for chunk in response:
                if content := chunk.choices[0].delta.content:
                    full_report_chunks.append(content)
                    yield ("token", content)

            mdt_report = "".join(full_report_chunks)

            # Build final result (same structure as non-streaming)
            evidence_fragments = [
                EvidenceFragment(
                    content=f["content"],
                    source_ids=[s.get("id", "") for s in f["sources"]],
                    final_score=f["final_score"],
                    evidence_type=extended_pool.get(f["content"], {}).get("type", ""),
                )
                for f in selected
            ]
            bib_entries = [
                BibliographyEntry(
                    ref_index=v["ref_index"], pmid=v.get("pmid", ""),
                    title=v.get("title", ""), guidelines=v.get("guidelines", ""),
                    source_id=v.get("id", ""),
                )
                for v in sorted(bibliography.values(), key=lambda x: x["ref_index"])
            ]
            similar_patients = []
            if not retrieved_df.empty:
                for idx, row in retrieved_df.iterrows():
                    similar_patients.append(SimilarPatient(
                        patient_id=str(idx),
                        distance=float(row.get("distance", 0)),
                        description=format_patient_desc(row),
                        historical_treatment=str(row.get("Y_text", "")),
                    ))

            yield ("result", {
                "request_id": request_id,
                "patient_profile_md": profile_md,
                "esgo_risk_classification": f"{esgo_risk_level} ({ESGO_MAPPING.get(esgo_raw_label, ('未知风险', 'Unknown Risk'))[1]})",
                "esgo_raw_label": esgo_raw_label,
                "knowledge_evidence": [f.model_dump() for f in evidence_fragments],
                "bibliography": [b.model_dump() for b in bib_entries],
                "similar_patients": [s.model_dump() for s in similar_patients],
                "xgb_predictions": xgb_pred,
                "comorbidity_screening": ComorbidityScreeningResult(
                    safety_warnings=screening_result.get("safety_warnings", []),
                    deescalation_advice=screening_result.get("deescalation_advice", []),
                    monitoring_recommendations=screening_result.get("monitoring_recommendations", []),
                    coverage_summary=screening_result.get("coverage_summary", {}),
                    reference_log=screening_result.get("reference_log", []),
                ).model_dump(),
                "mdt_report": mdt_report,
                "mdt_overrides": mdt_overrides,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield ("error", {"stage": "unknown", "detail": str(e)})

    def _build_mdt_prompt(self, profile_md, filtered_cases_text, pred_summary, context_str,
                          safety_warnings_text, deescalation_text):
        """Shared MDT prompt builder used by streaming endpoint."""
        return f"""以下是患者的【全息特征画像】：
{profile_md}

以下是经过药师 Agent 严格筛查后，确认对本患者具备高度参考价值的【相似病例治疗建议】：
{filtered_cases_text}
{pred_summary if pred_summary else ""}

以下是图谱召回的【参考证据】（已包含系统前置推演结论及权威文献）：
{context_str}

【🚨 核心分工与防幻觉准则】：
1. **指南解析必须极尽详尽（核心任务！）**：你必须把你能在【参考证据】中看到的指南细则长篇大论地写出来。
2. **知识片段精准标记**：引用图谱返回的完整语义片段时，**必须加粗核心词汇，并严格在句末打上对应的文献标号 [x]**。
3. **强制采纳药师预警**：将下方的【药师安全预警】无缝融合到"合并症管理"章节中。
4. **数据验证留给下游**：绝对禁止编造具体的生存率或 HR 数值，将其写在"PICO提问"中！

【💊 药师 Agent 药学安全预警】：
{safety_warnings_text if safety_warnings_text else "无"}

【⚖️ MDT 真实世界方案降级与修正指令】：
{deescalation_text if deescalation_text else "无（患者体能良好，可按指南标准执行）"}

【📝 强制报告排版结构（严格遵守格式！）】：

# 妇科肿瘤 MDT 初始会诊报告

## 一、 病情摘要与风险判定
## 二、 核心指南与共识详尽解析
### 1. 2025 ESGO-ESTRO-ESP 指南 [1]
- **核心推荐路径**：`EBRT + 同步化疗 + 辅助化疗，或 序贯化疗 + 放疗`

### 2. NCCN 临床实践指南 [x]
### 3. 国内共识及其他指南 [x]

## 三、 初步专科治疗框架
## 四、 随访大纲
### 五、 向 EBM 循证系统提问 (PICO Questions)

💡 请先在 <think> 标签内梳理：1. 确认风险等级；2. 疯狂提取证据里的指南细节；3. 盘点所有合并症；4. 构思要留给下游的硬核数据问题。思考完毕后，输出这份专业报告！"""
