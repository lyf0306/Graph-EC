import os
import asyncio
import numpy as np
import re
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import AsyncOpenAI
from pathorag_core import PathoRAG, QueryParam
from pathorag_core.utils import wrap_embedding_func_with_attrs
from pathorag_core.hyper_attention import init_attention_system

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax
import sys
from pathlib import Path

# ================= 模块路径配置 =================
# ================= 路径配置（环境变量优先，回退至项目相对路径） =================
_PROJECT_ROOT = Path(__file__).parent.parent
CLUSTER_RAG_DIR = Path(os.environ.get("CLUSTER_RAG_DIR", _PROJECT_ROOT / "CLUSTER_RAG_Endometrial"))
SRC_DIR = CLUSTER_RAG_DIR / "src"

for p in [str(CLUSTER_RAG_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
        
# ================= 导入 V4 核心逻辑 =================
from v4_llm_pipeline import (
    build_prompt as build_pathology_prompt,
    extract_json_robust,
    normalize_X,
    get_base_stage_2023_from_2009
)
from v4_comorbidity_extractor import build_comorbidity_prompt, COMORBIDITY_SCHEMA
from v4_esgo_decision_tree import classify_esgo_risk
from utils.trans_format import format_patient_desc

# ================= 合并症筛查 Skill（Agent 模式） =================
from comorbidity_skill import ComorbidityScreeningSkill

# ================= v4 数据与模型路径 =================
V4_PREPROCESSOR_PATH = CLUSTER_RAG_DIR / "models" / "preprocessor_retriever_v4.pkl"
V4_KMEANS_PATH = CLUSTER_RAG_DIR / "models" / "kmeans_retriever_v4.pkl"
V4_FEATURES_PATH = CLUSTER_RAG_DIR / "models" / "X_vec_retriever_v4.npy"
V4_KNN_PATH = CLUSTER_RAG_DIR / "models" / "knn_index_retriever_v4.pkl"
V4_IDS_PATH = CLUSTER_RAG_DIR / "models" / "patient_ids_retriever_v4.pkl"
V4_DF_PATH = CLUSTER_RAG_DIR / "models" / "df_retriever_v4.pkl"
V4_MODEL_PATH = CLUSTER_RAG_DIR / "models" / "trained_xgb_v4.pkl"

THRESHOLDS = {
    'radiotherapy': 0.5, 'chemotherapy': 0.5, 'targeted_therapy': 0.2,
    'immunotherapy': 0.2, 'hormone_therapy': 0.3
}

# ================= API 配置 =================
DEEPSEEK_API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-chat"  

RERANK_API_URL = "http://localhost:8001/v1"
RERANK_API_KEY = "EMPTY"
RERANK_MODEL_NAME = "QwenReranker"

EMBEDDING_API_URL = "http://localhost:8002/v1"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_MODEL_NAME = "QwenEmbedding"
EMBEDDING_DIM = 2560

WORKING_DIR = os.environ.get("WORKING_DIR", str(_PROJECT_ROOT / "pathorag_core" / "working"))
MOE_MODEL_PATH = os.environ.get("MOE_MODEL_PATH", str(_PROJECT_ROOT / "result" / "moe_router.pth"))

os.environ.setdefault("NEO4J_URI", "neo4j://localhost:7688")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "")
os.environ["NEO4J_DATABASE"] = "neo4j"
os.environ["MILVUS_URI"] = "http://localhost:19530"

init_attention_system(
    model_path=os.environ.get("ATTENTION_MODEL_PATH", str(_PROJECT_ROOT / "result" / "clinical_attention_v3.pth")),
    vdb_path=os.environ.get("VDB_ENTITIES_PATH", str(_PROJECT_ROOT / "result" / "vdb_entities.json")),
    embedding_dim=2560
)

# ================= 参考文档路径 =================
NCCN_REFERENCE_PATH = SRC_DIR.parent / "references" / "NCCN_2026v1.md"

# ================= MoE 门控路由器网络 =================
class MoERouter(nn.Module):
    def __init__(self, input_dim, temperature_init=0.5):
        super(MoERouter, self).__init__()
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_guideline_tier(guideline_str):
    if not guideline_str: return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str: return 1
    elif "FIGO" in g_str: return 2
    elif "NCCN" in g_str: return 3
    else: return 4

async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, 'driver'): return sources_info
        async with storage.driver.session() as session:
            if knowledge_text.startswith("【权威循证溯源："):
                match = re.search(r'【权威循证溯源：(.*?)】', knowledge_text)
                if match:
                    paper_name = match.group(1).strip()
                    cypher_query = """
                    MATCH (paper:Paper)
                    WHERE paper.name = $paper_name OR paper.pmid = $paper_name
                    RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                    """
                    result = await session.run(cypher_query, paper_name=paper_name)
                    records = await result.data()
                else: records = []
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
                src_id = record.get("src_id", "Unknown").replace('"', '')
                raw_pmid = record.get("pmid")
                final_pmid = str(raw_pmid) if (raw_pmid and len(str(raw_pmid)) < 20) else src_id.replace("paper::", "") if "paper::" in src_id else "Unknown"
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                sources_info.append({"id": src_id, "pmid": final_pmid, "title": record.get("title") or "No Title", "guidelines": gl_str})
    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    return sources_info

# ================= 🚀 药师、质控与 MDT 评估 Agent =================
# (已迁移至 ComorbidityScreeningSkill: 合并症筛查已升级为 Agent 模式)


# ================= 特征提取与辅助模块 =================
async def extract_patient_profile(patient_case, client):
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

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=2048, response_format={"type": "json_object"}
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
        # LLM 未提取手术信息时，从原始病历正则回退提取
        if not procedures_list:
            import re as _re
            _m = _re.search(r'手术方式及术中所见：(.+?)(?:\n\n|\n\d)', patient_case, _re.DOTALL)
            if _m:
                _txt = _m.group(1)
                procedures_list = [p.strip() for p in _re.split(r'[、，]', _txt.split('\n')[0]) if p.strip()]
                surgery["procedures"] = procedures_list
                # 提取术中所见
                _lines = _txt.strip().split('\n')
                if len(_lines) > 1:
                    surgery["intraoperative_findings"] = " ".join(l.strip() for l in _lines[1:] if l.strip())[:300]
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
        print(f"  -> 成功提取全息患者画像与检索词。")
        return profile_md, bilingual_keywords
    except Exception as e:
        print(f"[Warning] 提取特征失败: {e}")
        return patient_case, ""

async def get_structured_x_features(patient_case: str, llm_client):
    pathology_p = build_pathology_prompt("realtime_task", patient_case)
    comorbidity_p = build_comorbidity_prompt("realtime_task", patient_case)
    tasks = [
        llm_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": pathology_p}], response_format={"type": "json_object"}),
        llm_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": comorbidity_p}], response_format={"type": "json_object"})
    ]
    patho_res, como_res = await asyncio.gather(*tasks)
    patho_json = extract_json_robust(patho_res.choices[0].message.content)
    como_json = extract_json_robust(como_res.choices[0].message.content)
    new_patient_dict = normalize_X(patho_json.get("X", patho_json))
    
    stage_2023_val, _ = get_base_stage_2023_from_2009(
        new_patient_dict.get("stage_raw", "I"), new_patient_dict.get("histology_type", "serous"),
        new_patient_dict.get("grade", "G1"), new_patient_dict.get("myometrial_invasion_ratio", "<50%"),
        new_patient_dict.get("myometrial_invasion_depth", None), new_patient_dict.get("lvsi", "negative"),
        new_patient_dict.get("lvsi_substantial", False), new_patient_dict.get("cervical_involvement", "negative"),
        new_patient_dict.get("adnexal_involvement", 0), new_patient_dict.get("lymph_node_pelvic", "0/0"),
        new_patient_dict.get("lymph_node_paraaortic", "negative"), new_patient_dict.get("peritoneal_cytology", "negative")
    )
    new_patient_dict["stage_2023"] = stage_2023_val
    como_x = como_json.get("X", como_json)
    for key in COMORBIDITY_SCHEMA.keys():
        new_patient_dict[key] = como_x.get(key, 0)
    new_patient_dict["esgo_risk_group"] = classify_esgo_risk(new_patient_dict)
    return new_patient_dict

async def compute_rerank_score(query, doc, client):
    instruction = "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan."
    prompt = f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    try:
        response = await client.completions.create(model=RERANK_MODEL_NAME, prompt=prompt, max_tokens=1, temperature=0,
                                                   logprobs=20)
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        true_logit, false_logit = -10.0, -10.0
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes": true_logit = max(true_logit, logprob)
            elif clean_token == "no": false_logit = max(false_logit, logprob)
        true_score, false_score = math.exp(true_logit), math.exp(false_logit)
        return 0.0 if true_score + false_score == 0 else true_score / (true_score + false_score)
    except Exception: return 0.0

def compute_soft_cluster_features(X_base, kmeans):
    distances = kmeans.transform(X_base)
    return softmax(-distances, axis=1)

class PatientRetrieverV4:
    def __init__(self):
        self.preprocessor = joblib.load(V4_PREPROCESSOR_PATH)
        self.kmeans = joblib.load(V4_KMEANS_PATH)
        self.knn = joblib.load(V4_KNN_PATH)
        self.df = pd.read_pickle(V4_DF_PATH)
        self.patient_ids = joblib.load(V4_IDS_PATH)
        self.X_vec = np.load(V4_FEATURES_PATH)
        self.classifiers = joblib.load(V4_MODEL_PATH) if V4_MODEL_PATH.exists() else None
        self.label_names = ['radiotherapy', 'chemotherapy', 'targeted_therapy', 'immunotherapy', 'hormone_therapy']

    def _prepare_new_patient_df(self, new_patient_dict):
        CATEGORICAL_COLS = ["X_stage_raw", "X_figo_version", "X_histology_type", "X_grade", "X_cervical_involvement", "X_menopause", "X_p53", "X_mmr", "X_molecular_subtype", "X_stage_2023", "X_esgo_risk_group", "X_myometrial_invasion_ratio", "X_lvsi", "X_peritoneal_cytology"]
        NUMERICAL_COLS = ["X_age", "X_myometrial_invasion_depth"]
        COMORBIDITY_COLS = ["X_glycemic_status", "X_hypertension", "X_bmi_status", "X_hyperlipidemia", "X_anemia", "X_hepatic_viral", "X_hepatic_dysfunction", "X_major_cv_risk"]
        OTHER_BINARY_COLS = ["X_lvsi_substantial", "X_adnexal_involvement"]
        WEIGHT_COLS = ["X_major_cv_risk", "X_hepatic_viral"]
        WEIGHT_MULTIPLIER = 2.0

        expected_cols = self.preprocessor.feature_names_in_
        full_dict = {}
        for col in expected_cols:
            raw_name = col[2:] 
            if raw_name in new_patient_dict: full_dict[col] = new_patient_dict[raw_name]
            else:
                if col in NUMERICAL_COLS: full_dict[col] = np.nan
                elif col in COMORBIDITY_COLS or col in OTHER_BINARY_COLS: full_dict[col] = 0
                else: full_dict[col] = 'unknown'

        df = pd.DataFrame([full_dict])
        for col in CATEGORICAL_COLS:
            if col in df.columns: df[col] = df[col].astype(str)
        for col in NUMERICAL_COLS:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        for col in COMORBIDITY_COLS + OTHER_BINARY_COLS:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        for col in WEIGHT_COLS:
            if col in df.columns: df[col] = df[col] * WEIGHT_MULTIPLIER
        for col in COMORBIDITY_COLS + OTHER_BINARY_COLS:
            if col in df.columns and col not in WEIGHT_COLS: df[col] = df[col].astype(int)
        return df

    def _transform_patient(self, patient_df):
        X_base = self.preprocessor.transform(patient_df)
        distances = self.kmeans.transform(X_base)
        X_soft = softmax(-distances, axis=1)
        return np.column_stack([X_base, X_soft])

    def retrieve(self, new_patient_dict, top_k=3):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self._transform_patient(new_df)
        dist, idx = self.knn.kneighbors(new_vec, n_neighbors=top_k)
        results = self.df.iloc[idx[0]].copy()
        results['distance'] = dist[0]
        return results

    def format_similar_cases_for_prompt(self, new_patient_dict, top_k=3, verbose=True):
        retrieved_df = self.retrieve(new_patient_dict, top_k=top_k)
        lines = ["\n【真实世界相似病例参考（历史治疗方案）】"]
        for i, (idx, row) in enumerate(retrieved_df.iterrows()):
            order = ["最相似病例", "次相似病例", "第三相似病例"][i]
            dist = row.get('distance', 'N/A')
            lines.append(f"\n{order} (ID: {idx}, 距离: {dist:.4f}):")
            lines.append(format_patient_desc(row))
            y_text = row.get('Y_text', '')
            if y_text and y_text.strip() and y_text != 'unknown':
                lines.append(f"  历史治疗建议原文: {y_text}")
        return "\n".join(lines)

    def predict(self, new_patient_dict):
        if self.classifiers is None: return {}
        new_df = self._prepare_new_patient_df(new_patient_dict)
        X_final = self._transform_patient(new_df) 
        predictions = {}
        for label in self.label_names:
            if label not in self.classifiers: continue
            clf = self.classifiers[label]
            proba = float(clf.predict_proba(X_final)[:, 1][0])
            thr = THRESHOLDS.get(label, 0.5)
            predictions[label] = 1 if proba > thr else 0
        return predictions


# ================= 主函数 =================
async def main():
    print(f">>> [1/5] 正在配置 API 客户端...")
    try:
        embed_client = AsyncOpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)
        rerank_client = AsyncOpenAI(base_url=RERANK_API_URL, api_key=RERANK_API_KEY)

        @wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
        async def embedding_func(texts):
            if isinstance(texts, str): texts = [texts]
            response = await embed_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
            return np.array([data.embedding for data in response.data])

        print("API 客户端配置成功。")
    except Exception as e:
        print(f"API 客户端配置失败: {e}")
        exit(1)

    async def vector_stream_reranker(query_str, docs_list):
        if not docs_list: return []
        scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
        return [doc for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True) if score > 0.01]

    print(">>> [加载模型] 正在挂载 MoE 门控路由器...")
    moe_model = MoERouter(input_dim=EMBEDDING_DIM).to(DEVICE)
    try:
        checkpoint = torch.load(MOE_MODEL_PATH)
        moe_model.load_state_dict(checkpoint, strict=False)
        moe_model.eval()
        print("✅ MoE 路由器挂载成功！")
    except Exception as e:
        print(f"❌ MoE 路由器挂载失败，请检查路径: {e}")
        exit(1)

    print(f">>> [2/5] 正在初始化 PathoRAG 检索引擎...")
    try:
        graph_engine = PathoRAG(
            working_dir=WORKING_DIR, embedding_func=embedding_func,
            kv_storage="JsonKVStorage", vector_storage="MilvusVectorDBStorge",
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"PathoRAG 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)

    # 原始长病历数据
    patient_case = (
        """# 患者病情：
## 现病史:
患者为绝经后女性，因"绝经后阴道少量出血1月"就诊。患者绝经10年，绝经前月经规律，平素无特殊不适。就诊后行妇科超声提示子宫内膜增厚伴宫腔占位，行宫腔镜检查及诊断性刮宫术，病理提示子宫内膜样腺癌。为进一步手术治疗收入院。

## 既往史:
有高血压病史5年，规律口服降压药物，血压控制可。否认糖尿病、冠心病、肝炎、结核等病史。否认手术外伤史。否认输血史。无食物药物过敏史。

## 家族史:
否认家族遗传病史及肿瘤史。
体格检查：生命体征平稳，一般情况可。妇科检查：外阴正常，阴道通畅，宫颈光滑，子宫前位，大小正常，活动可，双附件区未及明显异常。

## 主要辅助检查：
【肿瘤标志物】（XX年XX月XX日）: CA125 及 HE4 轻度升高，余未见明显异常。
【（病理）小标本】（XX年XX月XX日） 诊断描述 （宫腔占位）子宫内膜样腺癌，II级。免疫组化结果：ER（+，80%），PR（+，50%），p53（野生表型），Ki-67（+，30%），MLH1（+），MSH2（+），MSH6（+），PMS2（+）。
【[MR]盆腔平扫+增强】（XX年XX月XX日） 检查所见 子宫增大，宫腔内见占位性病变，T2WI呈稍高信号，DWI弥散受限，肌层浸润约1/2，宫旁及盆壁未见明确浸润，盆腔淋巴结未见明显肿大。
【肿瘤标志物,性激素】（XX年XX月XX日）: 癌胚抗原 1.71 ng/ml, 甲胎蛋白定量 1.49 ng/ml, 糖类抗原CA125 117.89 U/ml↑, 人附睾蛋白4 84.3 pmol/L↑。
【（病理）小标本】（XX年XX月XX日） 诊断描述 （宫颈赘生物）低分化腺癌，结合免疫组化为伴p53突变的高级别子宫内膜癌。免疫组化结果：CK8/18（+），ER（+，90%，中-强），PR（-），p53（突变型/过表达），p16（弥漫强+），Ki-67（90%+），MLH1（完整），PMS2（完整），MSH2（完整），MSH6（完整）。
术前诊断：1. 子宫内膜恶性肿瘤 2. 高血压 3. 子宫多发性平滑肌瘤
手术方式及术中所见：XX年XX月XX日 行腹腔镜下全子宫切除术、腹腔镜下双侧卵巢和输卵管切除术、前哨淋巴结活组织检查、腹腔镜下盆腔粘连松解术。探查：腹水无；横膈、肝脾表面、肠系膜及结肠旁沟均无明显异常；子宫前位，增大，宫体见多发肌瘤样结节，最大者直径1cm，双附件与盆壁轻度粘连，未见明显异常。

## 术后病理：
（腹腔冲洗液）未见肿瘤细胞。
病理诊断：子宫内膜样腺癌，II级，浸润肌层约1/2，未见脉管癌栓，宫颈间质未累及。
淋巴结：（左前哨淋巴结）2枚，超分期未见癌转移，（0/2）。（右前哨淋巴结）2枚，超分期未见癌转移，（0/2）。
免疫组化：肿瘤细胞 CK7（+），PAX8（+），ER（+，90%，中），PR（-），p53（突变型/过表达），Ki-67（90%+），MLH1（完整），PMS2（完整），MSH2（完整），MSH6（完整）。
MMR免疫组化：正常。结论：无错配修复缺陷的免疫组化证据。
分子分型已查，结果未出。

# 术后诊断：
1. 子宫内膜恶性肿瘤 (子宫内膜样腺癌IIIA1期（FIGO 2023）/T3aN0M0（AJCC 8th）) 2. 子宫多发性平滑肌瘤 3. 高血压 4. 人乳头瘤病毒感染
"""
    )

    print(f"\n>>> [3/5] 输入原始病例:\n")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 提取全息患者画像与双语特征...")
    patient_profile_md, bilingual_keywords = await extract_patient_profile(patient_case, llm_client)

    # ============================================================
    # 👑 1. 特征字典与合并症筛查 Agent（Skill 模式）
    # ============================================================
    new_patient_dict = await get_structured_x_features(patient_case, llm_client)

    # 初始化合并症筛查 Skill（将单一 LLM 调用升级为 Agent）
    comorbidity_skill = ComorbidityScreeningSkill(
        llm_client=llm_client,
        reference_path=NCCN_REFERENCE_PATH,
    )

    # ============================================================
    # 🚀 新增：真实世界 MDT 降级与修正字典 (MDT Override)
    # ============================================================
    MDT_REAL_WORLD_OVERRIDES_DICT = {
        "major_cv_risk": "【真实世界MDT考量-严重心血管并发症/心脏支架】：患者存在极高心血管事件风险。若评估无法耐受大范围盆腔放疗或强效联合化疗的生理负荷，绝对不能死板生搬指南。建议启动MDT，强烈考虑【取消放疗】或【化疗降级/减药】，转向姑息性或单药低毒方案，以维持生活质量（QoL）为主。",
        "hepatic_dysfunction": "【真实世界MDT考量-器官功能受损】：患者存在肝脏或多脏器功能异常，对系统性化疗耐受极差。建议放弃标准剂量的高毒性化疗，全面转向减量方案或最佳支持治疗（BSC）。"
    }
    
    patient_mdt_overrides = []
    # 1. 检查是否存在严重心血管风险
    if new_patient_dict.get("major_cv_risk", 0) > 0 or new_patient_dict.get("major_cv_risk") is True:
        patient_mdt_overrides.append(MDT_REAL_WORLD_OVERRIDES_DICT["major_cv_risk"])
        
    # 2. 检查是否高龄衰弱 (利用动态参数触发：假设年龄>=75岁判定为高龄脆弱)
    try:
        age_val = float(new_patient_dict.get("age", 0))
        if age_val >= 75:
            patient_mdt_overrides.append("【真实世界MDT考量-高龄与极度衰弱】：患者为75岁以上高龄，预期ECOG评分较差。常规NCCN推荐的系统性强放化疗可能带来弊大于利的毒性。临床决策应从‘延长生存’向‘维持生活质量’倾斜，允许对标准方案进行减量、延迟或放弃放疗。")
    except:
        pass # 若年龄解析失败则跳过
        
    mdt_overrides_str = "\n".join(patient_mdt_overrides) if patient_mdt_overrides else "无特殊真实世界MDT降级考量。"

    # ============================================================
    # 👑 2. 直接使用本地 v4_esgo_decision_tree 算法结果定级
    # ============================================================
    print("\n>>> [3.6/5] 正在调用本地规则引擎进行 ESGO 2025 精准风险定级...")
    esgo_raw_label = new_patient_dict.get("esgo_risk_group", "unknown").lower()
    
    esgo_mapping = {
        "low": ("低危", "Low Risk"),
        "intermediate": ("中危", "Intermediate Risk"),
        "high-intermediate": ("中高危", "High-Intermediate Risk"),
        "high": ("高危", "High Risk"),
        "advanced": ("晚期转移", "Advanced Stage"),
        "unknown": ("不确定风险", "Uncertain Risk")
    }
    
    esgo_risk_level, en_risk_keyword = esgo_mapping.get(esgo_raw_label, ("未知风险", "Unknown Risk"))
    safe_clinical_conclusion = f"综合患者特征，经内置 ESGO 2025 算法严格计算判定为：【{esgo_risk_level} ({en_risk_keyword})】。"
    print(f"  -> ✅ [算法推演成功]: {safe_clinical_conclusion}")

    alias_mapping = {
        "Stage IA": "IA", "Stage IB": "IB", "Stage IIIA": "IIIA", "Stage IIIA1": "IIIA1",
        "I级": "G1", "II级": "G2", "III级": "G3", "高分化": "G1", "中分化": "G2", "低分化": "G3",
        "MMR正常": "pMMR", "MMR缺失": "dMMR", "阴性": "-", "阳性": "+"
    }

    pure_kws = []
    if bilingual_keywords:
        for line in bilingual_keywords.split('\n'):
            line = line.strip()
            if not line: continue
            if ']' in line and ':' in line: content = line.split(':', 1)[1]
            else: content = line
            parts = re.split(r'[,，]', content)
            pure_kws.extend([p.strip() for p in parts if p.strip()])

    if esgo_risk_level: pure_kws.append(esgo_risk_level.strip())
    if en_risk_keyword: pure_kws.append(en_risk_keyword.strip())

    seen = set()
    final_kws = []
    for kw in pure_kws:
        kw = kw.strip()
        if not kw: continue
        kw = re.sub(r'(?i)^Stage\s*', '', kw)
        kw = re.sub(r'期$', '', kw)
        mapped_kw = alias_mapping.get(kw, kw)
        if mapped_kw not in seen:
            seen.add(mapped_kw)
            final_kws.append(mapped_kw)

    graph_query = ", ".join(final_kws) if final_kws else patient_case[:200]
    vector_query = patient_case

    print(f"\n🚀 [图谱侧 Query (实体对齐后)]:\n{graph_query}")
    print("\n>>> [3.8/5] 🧠 正在通过 MoE 门控路由器分析患者病情，计算自适应融合权重...")
    try:
        query_emb = await embedding_func([graph_query])
        query_vec = query_emb[0] / (np.linalg.norm(query_emb[0]) + 1e-8)
        query_tensor = torch.tensor(query_vec, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            g_raw = moe_model(query_tensor.unsqueeze(0)).item()
            g = max(0.02, min(0.98, g_raw))
        print(f"  -> 🎯 MoE 动态门控计算完成！(图谱: {g:.4f}, 向量: {1.0-g:.4f})")
    except Exception as e:
        print(f"  -> ⚠️ MoE 权重计算失败，启用默认权重: {e}")
        g = 0.5

    print("\n>>> [4/5] 正在执行双路混合检索与融合...")
    extended_knowledge_pool = {}
    try:
        param = QueryParam(mode="hybrid", top_k=60, max_token_for_text_unit=4000, only_need_context=True)
        setattr(param, "moe_weight", g)

        retrieved_results = await graph_engine.aquery(graph_query, param)
        if isinstance(retrieved_results, list):
            for item in retrieved_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    if len(content) < 5 or content in ["RELATES_TO", "BELONG_TO", "EVIDENCE"]: continue
                    tag_info = "🧠 图谱高阶逻辑" if content.startswith("【权威循证溯源：") else "🧩 纯向量语义召回"
                    extended_knowledge_pool[content] = {"type": tag_info, "graph_score": graph_score}

            candidate_contents = list(extended_knowledge_pool.keys())
            llm_context_list = []
            if candidate_contents:
                semantic_scores = await asyncio.gather(*[compute_rerank_score(vector_query, c, rerank_client) for c in candidate_contents])
                fragments_to_sort = []
                for idx, content in enumerate(candidate_contents):
                    info = extended_knowledge_pool[content]
                    sources = await get_source_details(graph_engine, content)
                    best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
                    final_score = (g * info["graph_score"]) + ((1 - g) * semantic_scores[idx]) + (4 - best_tier) * 0.1
                    fragments_to_sort.append({"content": content, "sources": sources, "final_score": final_score})

                fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
                selected_fragments = fragments_to_sort[:10]

                bibliography = {}
                ref_counter = 1
                llm_context_list = []

                # ESGO 注入
                esgo_paper_id = "paper::40744042"
                bibliography[esgo_paper_id] = {
                    "id": esgo_paper_id, "pmid": "40744042",
                    "title": "ESGO-ESTRO-ESP guidelines for the management of patients with endometrial carcinoma: update 2025",
                    "guidelines": "ESGO指南", "ref_index": ref_counter
                }
                llm_context_list.append(f"【来源文献: [{ref_counter}] | 证据级别: ESGO指南 (首选)】 {safe_clinical_conclusion}")
                ref_counter += 1

                # 👑 合并症安全审查（由 ComorbidityScreeningSkill 动态处理）
                # 将在获取相似病例后由 Agent 判断覆盖度，不满足则查参考文档

                for frag in selected_fragments:
                    source_indices = []
                    for src in frag["sources"]:
                        src_id = src['id']
                        if src_id not in bibliography:
                            src['ref_index'] = ref_counter
                            bibliography[src_id] = src
                            ref_counter += 1
                        source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
                    ref_tag = f"【来源文献: {', '.join(source_indices)}】" if source_indices else ""
                    llm_context_list.append(f"{ref_tag} {frag['content']}")

            context_str = "\n\n".join(llm_context_list)
    except Exception as e:
        print(f"检索出错: {e}")
        return

    print("\n>>> [4.5/5] 正在检索真实世界相似病例...")
    try:
        retriever = PatientRetrieverV4()
        if "stage_2023" not in new_patient_dict: new_patient_dict["stage_2023"] = new_patient_dict.get("stage_raw", "I")
        retrieved_df = retriever.retrieve(new_patient_dict, top_k=3)
        xgb_pred = retriever.predict(new_patient_dict)
        pred_summary = "\n".join([f"- {k}: {'推荐' if v == 1 else '不推荐'}" for k, v in xgb_pred.items()])
    except Exception as e:
        print(f"  -> ⚠️ 相似病例检索失败: {e}")
        retrieved_df, pred_summary = pd.DataFrame(), ""

    print("\n>>> [4.8/5] 🧑‍⚕️ 正在运行【合并症筛查 Agent (Skill 模式)】审查相似病例覆盖度，不满足则查参考文档...")
    try:
        screening_result = await comorbidity_skill.screen(
            patient_dict=new_patient_dict,
            patient_profile_md=patient_profile_md,
            similar_patients_df=retrieved_df,
            mdt_overrides_str=mdt_overrides_str,
        )
    except Exception as e:
        print(f"  -> ⚠️ 合并症筛查 Agent 运行失败: {e}")
        screening_result = {
            "filtered_cases": [], "safety_warnings": [],
            "deescalation_advice": [], "monitoring_recommendations": [],
        }
    
    retained_ids = [c["case_id"] for c in screening_result.get("filtered_cases", []) if c.get("is_retained", False)]
    filtered_cases_text = "\n【经过合并症筛查 Agent 严格质控后的真实世界相似病例】\n"
    if retained_ids and not retrieved_df.empty:
        for idx, row in retrieved_df.iterrows():
            if str(idx) in str(retained_ids):
                filtered_cases_text += f"\n✅ [采纳病例] ID: {idx}\n"
                filtered_cases_text += format_patient_desc(row) + "\n"
                filtered_cases_text += f"  历史治疗建议原文: {row.get('Y_text', '')}\n"
    else:
        filtered_cases_text += "⚠️ 警告：无可用相似病例，请完全依赖指南证据制定方案。\n"

    safety_warnings_text = "\n".join([f"⚠️ {w}" for w in screening_result.get("safety_warnings", [])])
    deescalation_text = "\n".join([f"🛑 {a}" for a in screening_result.get("deescalation_advice", [])])

    # 输出筛查摘要
    cov = screening_result.get("coverage_summary", {})
    print(f"  -> 合并症覆盖: {cov.get('covered', '?')} 项已覆盖, "
          f"{cov.get('reference_lookup_count', '?')} 项查参考文档 | "
          f"安全警告: {len(screening_result.get('safety_warnings', []))} 条")
    if screening_result.get("reference_log"):
        for entry in screening_result["reference_log"]:
            print(f"    - {entry['comorbidity']}: {'✅ 相似病例覆盖' if entry['source'] == 'similar_patients' else '📖 查参考文档'}")
    print(f"\n💊 [合并症筛查 Agent 安全预警]:\n{safety_warnings_text}\n")

    print("\n>>> [5/5] 启动 MDT 专家 Agent 生成最终会诊报告...")
    system_prompt = "你是一名严谨的妇科肿瘤 MDT 首席专家。你的任务是：基于知识图谱提供的证据，对【权威指南推荐】进行极其详尽的拆解和论述；同时，为下游的 EBM 系统提出需要深度查证的具体临床数据问题。"
    
    user_prompt = f"""以下是患者的【全息特征画像】：
{patient_profile_md}

以下是经过药师 Agent 严格筛查后，确认对本患者具备高度参考价值的【相似病例治疗建议】：
{filtered_cases_text}
{pred_summary if pred_summary else ""}

以下是图谱召回的【参考证据】（已包含系统前置推演结论及权威文献）：
{context_str}

【🚨 核心分工与防幻觉准则】：
1. **指南解析必须极尽详尽（核心任务！）**：你必须把你能在【参考证据】中看到的指南细则长篇大论地写出来。
2. **知识片段精准标记**：引用图谱返回的完整语义片段时，**必须加粗核心词汇，并严格在句末打上对应的文献标号 [x]**。
3. **强制采纳药师预警**：将下方的【药师安全预警】无缝融合到“合并症管理”章节中。
4. **数据验证留给下游**：绝对禁止编造具体的生存率或 HR 数值，将其写在“PICO提问”中！
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
（🚨 必须打破将所有指南混写的格式！请将检索到的【每一个指南/共识单独列出子标题】。在对该指南进行连贯的分析后，必须独立提炼出该指南针对此患者的“核心推荐路径”，并用代码块高亮显示。）
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

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            stream=True, temperature=0.1, max_tokens=8192
        )
        print("\n>>> 模型回复:\n")
        async for chunk in response:
            if content := chunk.choices[0].delta.content: print(content, end="", flush=True)

        print("\n\n" + "=" * 20 + " 参考文献 (References) " + "=" * 20)
        if bibliography:
            for details in sorted(bibliography.values(), key=lambda x: x['ref_index']):
                idx, pmid_val, paper_id = details['ref_index'], details.get('pmid', 'Unknown'), details['id']
                print(f"[{idx}] PMID: {pmid_val}" if pmid_val != 'Unknown' else f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                print(f"    Title: {details['title']}\n    Guidelines: {details['guidelines']}\n" + "-" * 10)
        else:
            print("（本次检索未关联到具体文献节点）")
        print("=" * 50)
    except Exception as e:
        print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())