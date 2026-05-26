import os
import json
import random
import re
import math
import asyncio
import logging
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI  # 🌟 引入异步大模型客户端

# 导入 PathoRAG 组件
from pathorag_core import PathoRAG
from pathorag_core.utils import wrap_embedding_func_with_attrs
from pathorag_core.base import QueryParam  
from pathorag_core.hyper_attention import init_attention_system

# ================= 1. 配置区域 =================
INPUT_FILE = "/root/result/clinical_dataset_v3.jsonl" 
OUTPUT_FILE = "/root/result/moe_training_data_v2_subgraph.json" 
WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 
ATTENTION_MODEL_PATH = "/root/Model/clinical_attention_v1.pth" 

EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"      
RERANK_MODEL_PATH = "/root/Model/Qwen3-Reranker-4B"         

# [Neo4j 配置]
os.environ["NEO4J_URI"] = "neo4j://localhost:7688"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ.setdefault("NEO4J_PASSWORD", "")

# 🌟 [大模型 API 裁判配置] (请替换为您实际使用的 API Key 和 Base URL)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")
LLM_BASE_URL = "https://api.deepseek.com/v1" # 例如 DeepSeek, Qwen 等兼容接口
LLM_MODEL_NAME = "deepseek-chat" 
# ===============================================

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

print(f"\n>>> 🚀 正在启动神经符号学数据洗脱管线 (LLM 裁判增强版)...")
print(f">>> 📦 正在加载 Embedding 与 Reranker 双引擎...")

embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
@wrap_embedding_func_with_attrs(embedding_dim=embed_model.get_sentence_embedding_dimension(), max_token_size=8192)
async def embedding_func(texts):
    if isinstance(texts, str): texts = [texts]
    return embed_model.encode(texts, normalize_embeddings=True)
rerank_model = CrossEncoder(RERANK_MODEL_PATH, trust_remote_code=True)

# 初始化异步 LLM 客户端
llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

print(f">>> 🕸️ 正在连接 PathoRAG 与 Neo4j 超图数据库...\n")
graph_engine = PathoRAG(
    working_dir=WORKING_DIR, embedding_func=embedding_func,
    kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage", graph_storage="Neo4JStorage", 
)
init_attention_system(
    model_path=ATTENTION_MODEL_PATH, vdb_path=os.path.join(WORKING_DIR, "vdb_entities.json"),
    embedding_dim=embed_model.get_sentence_embedding_dimension()
)

def calculate_semantic_score(text1, text2):
    if not text1 or not text2: return 0.0
    logit = rerank_model.predict([text1, text2])
    return round(1 / (1 + math.exp(-float(logit))), 4)

# 🌟 核心新增：大模型异步裁判函数
async def is_postop_plan_via_llm(plan_text):
    """调用大模型严格判定该方案是否属于术后/系统性治疗/随访"""
    prompt = f"""你是一个严谨的肿瘤学专家。请判断以下医疗方案是否包含或属于【术后治疗、辅助治疗、系统性内科治疗、放化疗、靶向免疫、内分泌治疗、或随访复查/观察】。
如果该方案【仅仅】是纯粹的外科手术操作细节（如全子宫切除、淋巴结清扫、大网膜切除等），而完全没有提及术后处理或内科随访，请输出 False。
如果是术后方案、内科方案、或包含随访复诊，请输出 True。

方案内容：{plan_text}

请注意，只能输出 True 或 False，不要输出任何其他字符。"""
    
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        res = response.choices[0].message.content.strip().lower()
        return 'true' in res
    except Exception as e:
        # 兜底机制：如果 API 失败，使用宽泛关键词兜底防止管线崩溃
        fallback_signals = ["术后", "辅助", "随访", "化疗", "放疗", "靶向", "免疫", "内分泌", "观察", "维持", "复查", "定期", "随诊", "无需"]
        return any(sig in plan_text for sig in fallback_signals)

async def get_hyperedge_vector(raw_pos, graph_engine):
    """带偏置的向量检索（粗筛）"""
    if not raw_pos: return raw_pos
    enhanced_query = raw_pos + " 术后辅助治疗、系统治疗、放化疗、靶向免疫、维持治疗、随访观察"
    try:
        results = await graph_engine.hyperedges_vdb.query(enhanced_query, top_k=5)
        if not results: return raw_pos
        
        # 粗筛信号词（扩充了复查、定期等早期词汇）
        postop_signals = ["术后", "辅助", "随访", "化疗", "放疗", "靶向", "免疫", "内分泌", "观察", "系统", "维持", "复查", "定期", "随诊", "无需"]
        best_match = results[0]
        
        for res in results:
            name = res.get("hyperedge_name") or res.get("name") or ""
            if any(sig in name for sig in postop_signals):
                best_match = res
                break
                
        edge_name = best_match.get("hyperedge_name") or best_match.get("name") or ""
        if edge_name: return edge_name.replace("<hyperedge>", "")
    except Exception: pass 
    return raw_pos

async def fetch_subgraph_features(hyperedge_name, neo4j_driver):
    if not hyperedge_name: return []
    query_name = hyperedge_name if hyperedge_name.startswith("<hyperedge>") else f"<hyperedge>{hyperedge_name}"
    cypher = """MATCH (e)-[r:RELATES_TO]->(h) WHERE h.name = $name RETURN e.name AS entity, r.role AS role, e.idf_weight AS idf"""
    features = []
    try:
        async with neo4j_driver.session() as session:
            res = await session.run(cypher, name=query_name)
            for rec in await res.data():
                features.append({"entity": rec["entity"], "role": rec["role"] or "CONTEXT", "idf_weight": float(rec["idf"]) if rec.get("idf") is not None else 1.0})
    except Exception: pass
    return features

async def mine_hard_negatives(anchor_cn, is_complex, neo4j_driver):
    if not anchor_cn: return []
    negatives = set()
    try:
        async with neo4j_driver.session() as session:
            if match := re.search(r'FIGO\s*([IVX]+)[A-C]*期', anchor_cn):
                tgt = "III期' OR e.name CONTAINS 'IV期" if match.group(1) == 'I' else "I期' OR e.name CONTAINS 'II期"
                for rec in await (await session.run(f"MATCH (e)-[:RELATES_TO {{role: 'RECOMMENDATION'}}]->(h) WHERE (e.name CONTAINS '{tgt}') AND h.name STARTS WITH '<hyperedge>' RETURN DISTINCT h.name AS neg LIMIT 3")).data(): negatives.add(rec["neg"].replace("<hyperedge>", ""))
            if is_complex:
                for disease in [c.strip() for c in anchor_cn.split(",")[-2:]]:
                    if disease and disease not in ["无", "None", "未提及"]:
                        for rec in await (await session.run("MATCH (e)-[:RELATES_TO {role: 'CONTRAINDICATION'}]->(h) WHERE e.name CONTAINS $d AND h.name STARTS WITH '<hyperedge>' RETURN DISTINCT h.name AS neg LIMIT 2", d=disease)).data(): negatives.add(rec["neg"].replace("<hyperedge>", ""))
    except Exception: pass
    neg_list = list(negatives)
    random.shuffle(neg_list)
    return neg_list[:3]

async def extract_real_graph_score(target_clean, coherence_map):
    best_score = 0.01
    if not target_clean: return best_score
    for k_text, c_score in coherence_map.items():
        if target_clean in k_text: return max(best_score, float(c_score))
        clean_k = k_text.split("]: ")[-1].strip() if "]: " in k_text else k_text.strip()
        sim = await asyncio.to_thread(calculate_semantic_score, target_clean, clean_k)
        
        # 🌟 降低门槛至 0.55，并使用开根号减轻图谱惩罚力度
        if sim > 0.4: 
            best_score = max(best_score, float(c_score) * (sim ** 0.5))
    return round(best_score, 4)

async def process_record(record, neo4j_driver, semaphore):
    async with semaphore:
        anchor_cn = record.get("anchor_cn", "")
        raw_pos = record.get("raw_positive", "")
        
        if "未提供" in raw_pos and ("随访" in raw_pos or "辅助" in raw_pos or "术后" in raw_pos): return []
        if "仅提及" in raw_pos and "手术" in raw_pos and len(raw_pos) < 30: return []
            
        pos_plan = await get_hyperedge_vector(raw_pos, graph_engine)
        pos_plan = pos_plan.replace("<hyperedge>", "").strip()
        
        # 🌟 熔断 2：大模型 API 精准裁判 (替代了原来的死板关键词判断)
        is_postop = await is_postop_plan_via_llm(pos_plan)
        if not is_postop:
            return [] # 大模型判定为纯手术废话，丢弃！

        # 🌟 意图注入：改为正向温和引导，防止图谱引擎罢工
        intent_prompt = anchor_cn + " [重点关注并推演该患者的：术后辅助治疗、放化疗、内科靶向或随访复查方案]"
        
        try:
            inference = await graph_engine.aquery(intent_prompt, param=QueryParam(top_k=50, mode="hybrid"))
            coherence_map = {item["<knowledge>"]: item["<coherence>"] for item in inference}
        except Exception: coherence_map = {}

        pos_graph = await extract_real_graph_score(pos_plan, coherence_map)
        if pos_graph < 0.1: return [] 

        hard_negs = await mine_hard_negatives(anchor_cn, record.get("is_complex", False), neo4j_driver)
        if not hard_negs: return []

        pos_semantic = await asyncio.to_thread(calculate_semantic_score, anchor_cn, pos_plan)
        pos_subgraph = await fetch_subgraph_features(pos_plan, neo4j_driver)
        
        instances = []
        for neg in hard_negs:
            neg_plan = neg.replace("<hyperedge>", "").strip()
            if not neg_plan: continue
            
            neg_semantic = await asyncio.to_thread(calculate_semantic_score, anchor_cn, neg_plan)
            neg_graph = 0.01
            for k, v in coherence_map.items():
                if neg_plan in k: neg_graph = max(neg_graph, float(v))
            
            neg_subgraph = await fetch_subgraph_features(neg_plan, neo4j_driver)
            
            # 基础模板
            base_instance = {
                "positive_plan": pos_plan,          
                "negative_plan": neg_plan,          
                "positive_subgraph": pos_subgraph,  
                "negative_subgraph": neg_subgraph,  
                "pos_semantic_score": pos_semantic,
                "neg_semantic_score": neg_semantic,
                "pos_graph_score": round(pos_graph, 4),   
                "neg_graph_score": round(neg_graph, 4)
            }
                
            # 🌟 数据翻倍 1：组装中文样本
            cn_instance = base_instance.copy()
            cn_instance["anchor_text"] = anchor_cn
            instances.append(cn_instance)
            
            # 🌟 数据翻倍 2：组装英文样本 (跨语言对齐增广)
            anchor_en = record.get("anchor_en", "")
            if anchor_en:
                en_instance = base_instance.copy()
                en_instance["anchor_text"] = anchor_en
                instances.append(en_instance)
                
        return instances

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件：{INPUT_FILE}")
        return

    neo4j_driver = AsyncGraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    records = [json.loads(line) for line in open(INPUT_FILE, 'r', encoding='utf-8') if line.strip()]
    
    # 因为加入了 LLM API 网络请求，可以适当放大并发量以跑满带宽
    semaphore = asyncio.Semaphore(40) 
    tasks = [process_record(rec, neo4j_driver, semaphore) for rec in records]
    
    final_dataset = []
    kept_count = 0
    discarded_count = 0
    
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="图谱推演 & LLM 裁判质检", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
    
    for t in pbar:
        res = await t
        if res and len(res) > 0:
            kept_count += 1
            final_dataset.extend(res)
            
            tqdm.write(f"\n{'='*65}")
            tqdm.write(f"🟢 [裁判放行] 挖掘出 {len(res)} 条数据 (含中英双语对齐)！")
            
            sample = res[0]
            display_dict = {
                "Anchor (患者)": sample["anchor_text"][:50] + "...",
                "Pos Plan (指南方案)": sample["positive_plan"],
                "Scores (图谱分)": f"Pos: {sample['pos_graph_score']} | Neg: {sample['neg_graph_score']}",
            }
            tqdm.write(json.dumps(display_dict, ensure_ascii=False, indent=2))
            tqdm.write(f"{'='*65}")
        else:
            discarded_count += 1
            
        pbar.set_postfix({"✅ 保留": kept_count, "❌ 熔断": discarded_count})
            
    await neo4j_driver.close()
    
    print(f"\n>>> 挖掘完成！共生成 {len(final_dataset)} 条黄金超图对比数据。")
    print(f">>> 数据存活率: {kept_count/len(tasks)*100:.1f}% (剔除纯手术噪音后，由英文 Augmentation 实现了翻倍补偿)")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())