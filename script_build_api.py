import os
import json
import time
import argparse
import asyncio
import pdfplumber  # 引入强大的 PDF 处理库
from openai import AsyncOpenAI
from graphr1 import GraphR1

# ---------------------------------------------
# 配置部分
# ---------------------------------------------
API_KEY = "sk-45a3b1bbcdc34df2a9805b7614ac951f" 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
DATA_DIR = "/root/Graph-R1/data_for_hypergraph"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------------------------------------------
# 核心增强：高质量 PDF 解析函数
# ---------------------------------------------
def parse_pdf_high_quality(file_path):
    """
    高质量解析 PDF，尝试保留表格结构并去除页眉页脚噪音。
    """
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 1. 尝试提取表格 (病理报告中常有表格)
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    # 将表格转换为 Markdown 格式，利于 LLM 理解
                    # 过滤掉 None 值
                    cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                    # 简单的 Markdown 表格构建
                    if cleaned_table:
                        header = " | ".join(cleaned_table[0])
                        separator = " | ".join(["---"] * len(cleaned_table[0]))
                        body = "\n".join([" | ".join(row) for row in cleaned_table[1:]])
                        table_texts.append(f"\n{header}\n{separator}\n{body}\n")

                # 2. 提取正文文本
                # x_tolerance 和 y_tolerance 可以控制字符合并的容错率，避免单词被切断
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                
                # 3. (可选) 简单的页眉页脚过滤 
                # 假设页眉页脚在页面上下 5% 的位置，且字数很少，可以根据实际情况调整
                # 这里暂时保留全文，因为病理报告每一行都很重要
                
                # 将表格内容插入到文本流中，或者直接追加
                # 这里我们选择将提取到的文本和表格文本合并
                if text:
                    full_text.append(text)
                if table_texts:
                    full_text.extend(table_texts)
                    
    except Exception as e:
        print(f"解析 PDF {file_path} 失败: {e}")
        return ""
        
    return "\n".join(full_text)

# ---------------------------------------------
# API 调用包装
# ---------------------------------------------
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

# ---------------------------------------------
# 提取逻辑
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

# ---------------------------------------------
# 主入口
# ---------------------------------------------
async def insert_knowledge(data_source, unique_contexts):
    rag = GraphR1(
        working_dir=f"expr/{data_source}",
        llm_model_func=my_api_llm_call,
        llm_model_name=MODEL_NAME,
        chunk_token_size=1200,  
        chunk_overlap_token_size=50,
        graph_storage="Neo4JStorage"
    )    
    await extract_knowledge(rag, unique_contexts)
    print(f"知识超图为 '{data_source}' 构建成功。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="DeepSeek_API_Graph")
    args = parser.parse_args()
    
    print(f"开始从 {DATA_DIR} 加载数据...")
    unique_contexts = []
    
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        exit(1)
        
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            # 处理 TXT
            if filename.endswith(".txt"):
                print(f"  正在加载 (TXT): {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    unique_contexts.append(f.read())
            
            # 处理 JSONL
            elif filename.endswith(".jsonl"):
                print(f"  正在加载 (JSONL): {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if "contents" in data: unique_contexts.append(data["contents"])
                        elif "text" in data: unique_contexts.append(data["text"])
            
            # --- 处理 PDF (新增) ---
            elif filename.endswith(".pdf"):
                print(f"  正在加载 (PDF): {filename}")
                content = parse_pdf_high_quality(file_path)
                if len(content) > 50: # 简单的有效性检查，忽略空文档
                    unique_contexts.append(content)
                else:
                    print(f"  警告: PDF {filename} 内容过短或无法解析")
            
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
        print(f"构建过程中发生错误: {e}")