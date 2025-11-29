import os
import json
import time
import argparse
import asyncio  # 导入 asyncio
import torch
# import pdfplumber  # <-- 移除 PDF 库

# 导入 transformers 相关库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig  # <-- 更改 1: 重新导入 BitsAndBytesConfig
)
from peft import PeftModel
from graphr1 import GraphR1

# ---------------------------------------------
# 步骤 1: 定义您的路径和常量 (不变)
# ---------------------------------------------
BASE_MODEL_PATH = "/root/deepseek-r1-32b"
ADAPTER_PATH = "/root/train_2025-10-11-09-11-27"
DATA_DIR = "/root/Graph-R1/data_for_hypergraph"

# ---------------------------------------------
# 步骤 2: 全局加载模型 (应用 4-bit 量化)
# ---------------------------------------------
print(f"正在从 {BASE_MODEL_PATH} 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)
# 修正 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"正在从 {BASE_MODEL_PATH} 加载基础模型 (使用 4-bit 量化)...") # <-- 更改 2: 更新日志

# 更改 3: 定义 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 保持 bfloat16 作为计算类型
    bnb_4bit_use_double_quant=True,
)
print("已配置 4-bit (NF4) 量化。")

# 加载基础模型 (应用量化)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,  # <-- 更改 4: 应用量化配置
    # torch_dtype=torch.bfloat16,   # <-- 更改 5: 移除 (已被 bnb_config 替代)
    device_map="auto",                 # <-- 自动分配到 GPU
    trust_remote_code=True
)

print(f"正在应用 Adapter: {ADAPTER_PATH}...")
# 应用 LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval() # 设置为评估模式
print("模型加载 (已量化) 并应用 Adapter 完毕。")


# ---------------------------------------------
# 步骤 3: 创建异步(async) LLM 包装函数 (不变)
# ---------------------------------------------
async def my_local_llm_call(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """
    这是一个异步包装器，它将 GraphR1 的异步调用
    转换为对我们 *同步* 的 model.generate() 方法的调用。
    
    关键：它直接处理 GraphR1 传入的 *原始提示词字符串*。
    """
    
    # 我们忽略 system_prompt 和 history，因为 GraphR1 的建图提示词
    # (PROMPTS["entity_extraction"]) 
    # 已经是一个包含所有指令的完整字符串。
    
    print(f"正在调用本地 LLM (输入长度: {len(prompt)})...")
    try:
        # 关键修复 #1：直接对原始 prompt 字符串进行分词，不使用聊天模板
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 同步的模型生成
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048, # 确保有足够空间生成块的实体
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False # 实体提取是确定性任务，不需采样
            )
        
        # 解码 (跳过输入部分)
        response_text = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
    except Exception as e:
        print(f"调用 local_model.generate 失败: {e}")
        response_text = ""  # 返回空字符串，让 GraphR1 处理失败

    # 关键修复 #2：释放事件循环
    # 因为 model.generate 是同步的，我们必须 await sleep(0)
    # 来将控制权交还给 asyncio，防止冻结 GraphR1 的异步框架。
    await asyncio.sleep(0.0) 
    
    print(f"本地 LLM 返回 (输出长度: {len(response_text)})")
    return response_text

# ---------------------------------------------
# 步骤 4: 知识提取函数 (改为 async) (不变)
# ---------------------------------------------
async def extract_knowledge(rag, unique_contexts):
    """
    异步处理所有文档的插入。
    """
    print(f"总计需要插入 {len(unique_contexts)} 个文档。")
    
    # 使用原始脚本中的分批逻辑
    batch_size = 50  # 保持与原脚本一致
    total_batches = (len(unique_contexts) + batch_size - 1) // batch_size
    print(f"总计插入批次: {total_batches} (每批 {batch_size} 个文档)")

    for i in range(0, len(unique_contexts), batch_size):
        batch_contexts = unique_contexts[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        print(f"--- 正在处理批次 {current_batch_num}/{total_batches} ---")
        
        retries = 0
        max_retries = 50
        while retries < max_retries:
            try:
                # 关键修改：调用 rag.ainsert() (异步插入)
                await rag.ainsert(batch_contexts)
                print(f"批次 {current_batch_num} 成功插入。")
                break  # 成功，跳出重试循环
            except Exception as e:
                retries += 1
                print(f"批次 {current_batch_num} 插入失败 (重试 {retries}/{max_retries}), 错误: {e}")
                await asyncio.sleep(10)  # 异步等待
        
        if retries == max_retries:
            print(f"批次 {current_batch_num} 插入失败，已达最大重试次数。跳过此批次。")
    
    print("所有数据批次处理完毕。")

# ---------------------------------------------
# 步骤 5: 主插入函数 (改为 async) (更新注释)
# ---------------------------------------------
async def insert_knowledge(data_source, unique_contexts):
    """
    配置 GraphR1 并启动知识提取。
    """
    rag = GraphR1(
        working_dir=f"expr/{data_source}",  # 输出目录
        
        # 关键修改 #1：
        llm_model_func=my_local_llm_call,    # <-- 使用我们上面定义的包装函数
        llm_model_name=BASE_MODEL_PATH,      # <-- 传递一个名称供日志记录
        
        # 关键设置:
        # 鉴于我们 *已经* 使用了 4-bit 量化, VRAM 压力较小。
        # chunk_token_size 决定了 GraphR1 每次发送给 LLM 的文本块大小。
        # 1200 是一个合理的值。如果 VRAM 依然不足, 可尝试减小此值 (例如 512 或 1024)。
        chunk_token_size=1200,               # <-- 更改 6: 更新了对 chunk_size 的注释
        chunk_overlap_token_size=50,
        graph_storage="Neo4JStorage"
    )    
    
    # 调用异步的 extract_knowledge
    await extract_knowledge(rag, unique_contexts)
    
    print(f"知识超图为 '{data_source}' 构建成功。")
    print(f"构建好的文件存放在: expr/{data_source}/")

# ---------------------------------------------
# 步骤 6: main 执行入口 (不变)
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="MyDeepSeekGraph_Quantized") # 更新了默认名称
    args = parser.parse_args()
    data_source = args.data_source
    
    print(f"开始从 {DATA_DIR} 加载数据...")
    unique_contexts = []
    
    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        exit(1)
        
    # 遍历您数据目录中的所有文件
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
                        if "contents" in data: # 检查 'contents' 键
                            unique_contexts.append(data["contents"])
                        elif "text" in data: # 备用 'text' 键
                            unique_contexts.append(data["text"])
            
            # --- PDF 处理逻辑已移除 ---
                
            else:
                print(f"  跳过: {filename} (非 .txt 或 .jsonl)")
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")

    if not unique_contexts:
        print("错误：在数据目录中未找到可处理的文档。")
        exit(1)
        
    print(f"成功加载了 {len(unique_contexts)} 个文档。")
    
    # 启动异步构建过程
    try:
        asyncio.run(insert_knowledge(data_source, unique_contexts))
    except Exception as e:
        print(f"知识图构建过程中发生致命错误: {e}")