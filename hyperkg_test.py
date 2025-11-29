import os
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from graphr1 import GraphR1, QueryParam

# =================配置区域=================
# 1. 设置模型路径 (根据你的环境修改)
BASE_MODEL_PATH = "/root/deepseek-r1-32b" 

# 2. 设置 Neo4j 连接信息 (必须与构建超图时一致)
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"  # 请替换为你的实际密码

# 3. 设置工作目录 (必须指向你构建图时使用的同一个目录，以便加载 KV 和 Vector 数据)
WORKING_DIR = "/root/Graph-R1/expr" # 请修改为你实际的输出目录
# =========================================

print(">>> 正在加载本地模型 (4-bit)...")
# 复制自 script_build.py 的加载逻辑
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print(">>> 模型加载完成。")

# 定义一个符合 GraphR1 接口要求的异步 LLM 函数
async def local_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # 忽略 GraphR1 传入的额外参数 (如 hashing_kv 等)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, # 根据需要调整生成的长度
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

async def main():
    # 测试问题
    question = "FIGO2023分期中是如何区分IC期和IIC期的？"
    
    print(f"\n{'='*20} 测试开始 {'='*20}")
    print(f"问题: {question}\n")

    # ---------------------------------------------------------
    # 1. 直接让模型回答 (无 RAG)
    # ---------------------------------------------------------
    print(f"\n>>> [1] 模型直接回答 (Direct Generation):")
    direct_answer = await local_llm_func(question)
    print("-" * 40)
    print(direct_answer)
    print("-" * 40)

    # ---------------------------------------------------------
    # 2. 初始化 GraphR1 并检索回答 (RAG)
    # ---------------------------------------------------------
    print(f"\n>>> [2] 正在初始化 GraphR1 (Neo4j)...")
    rag = GraphR1(
        working_dir=WORKING_DIR,
        graph_storage="Neo4jStorage", # 指定使用 Neo4j
        llm_model_func=local_llm_func, # 传入我们自定义的本地模型函数
        llm_model_name="local-deepseek", # 名字随意，用于缓存键
        # 确保其他参数与构建时兼容，通常保持默认即可，或者根据 script_build.py 调整 embedding_func
    )

    print(f">>> [2] GraphR1 检索并回答 (RAG Generation):")
    # 使用 query 方法，它会自动进行 关键词提取 -> 检索 -> 生成
    rag_answer = await rag.aquery(question, param=QueryParam(mode="hybrid")) # mode="hybrid" 混合检索
    
    print("-" * 40)
    print(rag_answer)
    print("-" * 40)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())