import os
import json
import asyncio
import numpy as np
import math
import re
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_async

# ================= 引入你的 PathoRAG 引擎 =================
from pathorag_core import PathoRAG, QueryParam
from pathorag_core.utils import wrap_embedding_func_with_attrs

# ================= 1. 服务与路径配置 =================
INPUT_JSONL = "/root/result/contrastive_training_dataset_final.jsonl"
OUTPUT_JSON = "training_data_prepared.json"

WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph"

# Neo4j 数据库配置
os.environ.setdefault("NEO4J_URI", "neo4j://localhost:7688")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")

# API 配置
RERANK_API_URL = "http://localhost:8001/v1"
RERANK_API_KEY = "EMPTY"
RERANK_MODEL_NAME = "QwenReranker"

EMBEDDING_API_URL = "http://localhost:8002/v1"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_MODEL_NAME = "QwenEmbedding" 
EMBEDDING_DIM = 2560

# ================= 2. 初始化真实 API 客户端与工具 =================
print(">>> [1/4] 正在初始化 Embedding 与 Rerank 客户端...")
embed_client = AsyncOpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)
rerank_client = AsyncOpenAI(base_url=RERANK_API_URL, api_key=RERANK_API_KEY)

@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def embedding_func(texts):
    if isinstance(texts, str): texts = [texts]
    response = await embed_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
    return np.array([data.embedding for data in response.data])

async def compute_real_rerank_score(query, doc, client):
    """直接复用 test_mlp.py 中的真实 QwenReranker 打分逻辑"""
    instruction = "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan."
    prompt = f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    try:
        response = await client.completions.create(model=RERANK_MODEL_NAME, prompt=prompt, max_tokens=1, temperature=0, logprobs=20)
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        true_logit, false_logit = -10.0, -10.0
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes": true_logit = max(true_logit, logprob)
            elif clean_token == "no": false_logit = max(false_logit, logprob)
        true_score, false_score = math.exp(true_logit), math.exp(false_logit)
        return 0.0 if true_score + false_score == 0 else true_score / (true_score + false_score)
    except Exception as e:
        print(f"Rerank API Error: {e}")
        return 0.0

# ================= 3. 图谱分数匹配逻辑 =================
def extract_graph_score(target_text, retrieved_results, default_miss_score=0.1):
    """
    在图谱召回的列表中寻找目标文本，提取真实的 coherence 分数。
    如果没找到（说明图谱未召回该方案或因禁忌症拦截），给予极低的分数作为惩罚。
    """
    clean_target = target_text.replace('"', '').replace("'", "").strip()
    if not retrieved_results or not clean_target:
        return default_miss_score

    for item in retrieved_results:
        if isinstance(item, dict):
            content = item.get('<knowledge>', '')
            # 采用子串匹配，因为图谱返回的可能是拼接了上下文的宏观推演文本
            if clean_target in content or content in clean_target:
                # 命中图谱召回，提取真实的图谱信息熵分数
                return float(item.get('<coherence>', default_miss_score))
    
    return default_miss_score

# ================= 4. 主流程：数据生成与打分 =================
async def generate_dataset():
    print(">>> [2/4] 正在加载并挂载真实的 PathoRAG 知识超图引擎...")
    try:
        graph_engine = PathoRAG(
            working_dir=WORKING_DIR, 
            embedding_func=embedding_func,
            kv_storage="JsonKVStorage", 
            vector_storage="NanoVectorDBStorage", 
            graph_storage="Neo4JStorage"
        )
    except Exception as e:
        print(f"❌ PathoRAG 初始化失败: {e}")
        return

    # 1. 读取原始 JSONL
    print(f">>> [3/4] 正在解析 {INPUT_JSONL} 并提取正样本库...")
    raw_data = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): raw_data.append(json.loads(line.strip()))
            
    all_positives = [item["positive"] for item in raw_data if item.get("positive")]
    flattened_triplets = []

    print(">>> [4/4] 正在使用大模型和图谱引擎计算真实分数 (这可能需要一些时间)...")
    # 为了避免并发过高压垮 Neo4j 和 Reranker，采用顺序异步处理
    for i, item in enumerate(raw_data):
        anchor = item["anchor"]
        positive = item["positive"]
        hard_negatives = item.get("hard_negatives", [])
        valid_negatives = [neg for neg in hard_negatives if neg != "无相关困难负样本"]

        print(f"\r处理进度: {i+1}/{len(raw_data)} | 当前 Anchor: {anchor[:20]}...", end="")

        # 【真实图谱查询】获取当前 Anchor 的全量图谱召回结果
        try:
            param = QueryParam(mode="hybrid", top_k=40)
            retrieved_results = await graph_engine.aquery(anchor, param)
        except Exception:
            retrieved_results = []

        # 获取正样本的真实分数
        pos_g_score = extract_graph_score(positive, retrieved_results, default_miss_score=0.1)
        pos_s_score = await compute_real_rerank_score(anchor, positive, rerank_client)

        # 构建负样本
        negatives_to_process = []
        if len(valid_negatives) > 0:
            # 使用真实的 Hard Negatives
            for neg in valid_negatives:
                negatives_to_process.append({"text": neg, "is_hard": True})
        else:
            # In-Batch Negative (批内随机负采样)
            import random
            random_neg = random.choice(all_positives)
            while random_neg == positive and len(all_positives) > 1:
                random_neg = random.choice(all_positives)
            negatives_to_process.append({"text": random_neg, "is_hard": False})

        # 计算每个负样本的真实分数并装库
        for neg_info in negatives_to_process:
            neg_text = neg_info["text"]
            is_hard = neg_info["is_hard"]
            
            # 【核心逻辑】真实 Hard Negative 的图谱分应该非常低，如果图谱没召回它，赋予 0.05 制造梯度
            default_neg_graph = 0.05 if is_hard else 0.1
            neg_g_score = extract_graph_score(neg_text, retrieved_results, default_miss_score=default_neg_graph)
            
            # 真实语义分
            neg_s_score = await compute_real_rerank_score(anchor, neg_text, rerank_client)

            flattened_triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": neg_text,
                "pos_graph_score": round(pos_g_score, 4),
                "pos_semantic_score": round(pos_s_score, 4),
                "neg_graph_score": round(neg_g_score, 4),
                "neg_semantic_score": round(neg_s_score, 4),
                "is_hard_negative": is_hard
            })

    # 将跑完真实分数的训练集保存
    print("\n\n>>> 正在保存带有真实分数的训练数据集...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(flattened_triplets, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 数据集构建彻底完成！共生成 {len(flattened_triplets)} 个高质量三元组。")
    print(f"请使用上一步提供的 2_train_moe_router.py 开始训练你的门控网络！")

if __name__ == "__main__":
    asyncio.run(generate_dataset())