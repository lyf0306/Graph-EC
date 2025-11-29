import os
import asyncio
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from graphr1 import GraphR1, QueryParam
from graphr1.utils import wrap_embedding_func_with_attrs

# ================= 配置区域 =================
# 1. 路径配置 (请务必确认与构建时一致!)
# 之前构建脚本默认输出目录是 expr/DeepSeek_QwenEmbed_Graph
WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 

# Embedding 模型路径
EMBEDDING_MODEL_PATH = "/root/Qwen3-Embedding-4B"

# LLM 模型路径 (DeepSeek-R1-Distill-Qwen-32B or 7B)
BASE_MODEL_PATH = "/root/deepseek-r1-7b" # 您日志中显示加载的是 7B

# 2. Neo4j 配置 (使用远程地址)
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_DATABASE"] = "neo4j"
# ===========================================

print(f">>> [1/5] 正在加载 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
# 加载 Embedding 模型
try:
    embed_model = SentenceTransformer(
        EMBEDDING_MODEL_PATH, 
        trust_remote_code=True,
        device="cuda"
    )
except Exception as e:
    print(f"Embedding 模型加载失败: {e}")
    exit(1)

# 动态获取维度
EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding 模型加载完毕，维度: {EMBEDDING_DIM}")

# 定义适配 GraphR1 的 Embedding 函数
@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def my_qwen_embedding(texts: list[str], **kwargs) -> np.ndarray:
    """
    GraphR1 专用的异步 Embedding 包装器 (查询模式)
    """
    # 这里的 prompt_name="query" 非常重要，Qwen 模型对 query 和 doc 有不同的指令
    embeddings = await asyncio.to_thread(
        embed_model.encode, 
        texts, 
        prompt_name="query", 
        convert_to_numpy=True, 
        show_progress_bar=False
    )
    return embeddings

# ================= LLM 设置 =================
print(f">>> [2/5] 正在加载 LLM: {BASE_MODEL_PATH} ...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        quantization_config=bnb_config, 
        device_map="cuda:0",
        trust_remote_code=True
    )
except Exception as e:
    print(f"LLM 加载失败: {e}")
    exit(1)

async def local_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.7,
            do_sample=True
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# ================= 主流程 =================
async def main():
    question = "FIGO2023分期中是如何区分IC期和IIC期的？"
    
    print(f"\n{'='*10} Graph-R1 推理测试 {'='*10}")
    print(f"工作目录: {WORKING_DIR}")
    print(f"问题: {question}")
    
    if not os.path.exists(WORKING_DIR):
        print(f"错误: 找不到工作目录 {WORKING_DIR}")
        return

    # [3/5] 初始化 GraphR1
    print("\n>>> [3/5] 初始化 GraphR1 检索引擎...")
    try:
        rag = GraphR1(
            working_dir=WORKING_DIR, 
            graph_storage="Neo4JStorage",
            llm_model_func=local_llm_func,
            embedding_func=my_qwen_embedding, # 必须传入以支持向量检索
            node2vec_params={
                "dimensions": EMBEDDING_DIM,
                "num_walks": 10, 
                "walk_length": 40
            }
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        return

   # [4/5] 执行向量检索
    print("\n>>> [4/5] 执行向量检索 (Vector Retrieval)...")
    try:
        top_k = 10
        # 1. 获取检索结果 (字典列表)
        entity_results = await rag.entities_vdb.query(question, top_k=top_k)
        hyperedge_results = await rag.hyperedges_vdb.query(question, top_k=top_k)
        
        # 2. 【关键】提取 Name 列表
        # 现在有了 meta_fields, 结果中会有 entity_name / hyperedge_name
        entity_match = [r['entity_name'] for r in entity_results if 'entity_name' in r]
        hyperedge_match = [r['hyperedge_name'] for r in hyperedge_results if 'hyperedge_name' in r]
        
        print(f"  - 检索到相关实体: {len(entity_match)} 个")
        # print(f"    {entity_match[:3]}...") # 打印前3个看看
        
    except Exception as e:
        print(f"向量检索失败: {e}")
        return

    # [5/5] 执行 GraphR1 查询
    print("\n>>> [5/5] 执行 GraphR1 上下文组装与生成...")
    try:
        response = await rag.aquery(
            question, 
            param=QueryParam(
                mode="hybrid", 
                top_k=5, 
                max_token_for_text_unit=4000
            ),
            entity_match=entity_match,      # 传入名字列表
            hyperedge_match=hyperedge_match # 传入名字列表
        )
        
        print("\n" + "="*20 + " 最终回答 " + "="*20)
        print(response)
        print("="*50)
        
    except Exception as e:
        print(f"查询生成过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())