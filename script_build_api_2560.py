import os
import json
import time
import argparse
import asyncio
import numpy as np
import pdfplumber
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from graphr1 import GraphR1
from graphr1.utils import wrap_embedding_func_with_attrs

# ---------------------------------------------
# 1. 配置路径与 API
# ---------------------------------------------
# LLM 配置 (DeepSeek API)
API_KEY = "sk-45a3b1bbcdc34df2a9805b7614ac951f" 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# Embedding 配置 (本地 Qwen 模型)
EMBED_MODEL_PATH = "/root/Qwen3-Embedding-4B"

# 数据目录 (保持您脚本中的路径)
DATA_DIR = "/root/Graph-R1/data_for_hypergraph"

# ---------------------------------------------
# 2. 初始化模型
# ---------------------------------------------

# A. 初始化 DeepSeek API 客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
print(f"LLM 客户端已就绪: {MODEL_NAME}")

# B. 初始化本地 Qwen Embedding 模型
print(f"正在加载 Embedding 模型: {EMBED_MODEL_PATH} ...")
try:
    # 尝试开启 flash_attention_2 以加速 (如果显卡支持)
    embed_model = SentenceTransformer(
        EMBED_MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"}
    )
except Exception as e:
    print(f"Flash Attention 加载失败，回退到默认模式: {e}")
    embed_model = SentenceTransformer(
        EMBED_MODEL_PATH, 
        trust_remote_code=True,
        device="cuda"
    )

# 动态获取模型维度 (确保维度参数绝对正确)
EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding 模型加载完毕，维度: {EMBEDDING_DIM}")

# ---------------------------------------------
# 3. 核心功能函数
# ---------------------------------------------

# --- 新增: 自定义 Embedding 函数 (适配 GraphR1 接口) ---
@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def my_qwen_embedding(texts: list[str], **kwargs) -> np.ndarray:
    """
    使用 Qwen-Embedding-4B 生成向量。
    """
    # 使用 asyncio.to_thread 将同步的 GPU 计算放入线程池，防止阻塞事件循环
    embeddings = await asyncio.to_thread(
        embed_model.encode, 
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False,
        batch_size=16 # 根据显存情况调整
    )
    return embeddings

# --- 原有: API 调用包装器 ---
async def my_api_llm_call(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        return content if content else ""
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""

# --- 原有: 高质量 PDF 解析器 ---
def parse_pdf_high_quality(file_path):
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 1. 尝试提取表格
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                    if cleaned_table:
                        header = " | ".join(cleaned_table[0])
                        separator = " | ".join(["---"] * len(cleaned_table[0]))
                        body = "\n".join([" | ".join(row) for row in cleaned_table[1:]])
                        table_texts.append(f"\n{header}\n{separator}\n{body}\n")

                # 2. 提取正文
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                
                if text: full_text.append(text)
                if table_texts: full_text.extend(table_texts)
                    
    except Exception as e:
        print(f"PDF 解析失败 {file_path}: {e}")
        return ""
    return "\n".join(full_text)

# ---------------------------------------------
# 4. 主构建逻辑
# ---------------------------------------------
async def extract_knowledge(rag, unique_contexts):
    print(f"总计需要插入 {len(unique_contexts)} 个文档。")
    batch_size = 50
    total_batches = (len(unique_contexts) + batch_size - 1) // batch_size
    
    for i in range(0, len(unique_contexts), batch_size):
        batch_contexts = unique_contexts[i:i + batch_size]
        print(f"--- 正在处理批次 {(i // batch_size) + 1}/{total_batches} ---")
        
        retries = 0
        while retries < 5:
            try:
                await rag.ainsert(batch_contexts)
                print(f"批次 {(i // batch_size) + 1} 成功插入。")
                break
            except Exception as e:
                retries += 1
                print(f"重试 {retries}/5: {e}")
                await asyncio.sleep(5)

async def insert_knowledge(data_source, unique_contexts):
    rag = GraphR1(
        working_dir=f"expr/{data_source}",
        
        # LLM 部分
        llm_model_func=my_api_llm_call,
        llm_model_name=MODEL_NAME,
        
        # --- 新增: Embedding 部分 ---
        embedding_func=my_qwen_embedding,
        
        # --- 新增: 维度同步 ---
        # 必须确保图嵌入(Node2Vec)的维度与文本嵌入维度一致
        node2vec_params={
            "dimensions": EMBEDDING_DIM, 
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        },
        
        # 其他配置
        chunk_token_size=1600,  
        chunk_overlap_token_size=50,
        graph_storage="Neo4JStorage"
    )    
    await extract_knowledge(rag, unique_contexts)
    print(f"知识超图为 '{data_source}' 构建成功。")

# ---------------------------------------------
# 5. 程序入口
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 建议更改 data_source 名称以避免与旧的 1536 维数据冲突
    parser.add_argument("--data_source", type=str, default="DeepSeek_QwenEmbed_Graph")
    args = parser.parse_args()
    
    print(f"开始从 {DATA_DIR} 加载数据...")
    unique_contexts = []
    
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        exit(1)
        
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            if filename.endswith(".txt"):
                print(f"  正在加载 (TXT): {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    unique_contexts.append(f.read())
            elif filename.endswith(".jsonl"):
                print(f"  正在加载 (JSONL): {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if "contents" in data: unique_contexts.append(data["contents"])
                        elif "text" in data: unique_contexts.append(data["text"])
            elif filename.endswith(".pdf"):
                print(f"  正在加载 (PDF): {filename}")
                content = parse_pdf_high_quality(file_path)
                if len(content) > 50: 
                    unique_contexts.append(content)
                else:
                    print(f"  警告: PDF {filename} 内容过短")
            else:
                print(f"  跳过: {filename}")
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")

    if not unique_contexts:
        print("错误：未找到有效文档。")
        exit(1)
        
    print(f"成功加载了 {len(unique_contexts)} 个文档。")
    
    try:
        asyncio.run(insert_knowledge(args.data_source, unique_contexts))
    except Exception as e:
        print(f"构建过程中发生致命错误: {e}")