import os
import asyncio
from pathlib import Path
import numpy as np
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import AsyncOpenAI
from pathorag_core import PathoRAG, QueryParam
from pathorag_core.utils import wrap_embedding_func_with_attrs

# ================= 配置区域 =================

# LLM 服务配置
VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "EMPTY" 
LLM_MODEL_NAME = "OriClinical" 

# Rerank 服务配置
RERANK_API_URL = "http://localhost:8001/v1"
RERANK_API_KEY = "EMPTY"
RERANK_MODEL_NAME = "QwenReranker"

# Embedding 服务配置
EMBEDDING_API_URL = "http://localhost:8002/v1"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_MODEL_NAME = "QwenEmbedding" 
EMBEDDING_DIM = 2560

_PROJECT_ROOT = Path(__file__).parent.parent
WORKING_DIR = os.environ.get("WORKING_DIR", str(_PROJECT_ROOT / "pathorag_core" / "working")) 

# Neo4j 数据库配置
os.environ.setdefault("NEO4J_URI", "neo4j://localhost:7688")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")

# ================= 深度学习打分网络 (尺寸已对齐 128) =================
class ClinicalScoringNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(ClinicalScoringNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 4, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128) 
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(128, 1)          

    def forward(self, anchor_emb, candidate_emb):
        diff = torch.abs(anchor_emb - candidate_emb)
        mult = anchor_emb * candidate_emb
        x = torch.cat([anchor_emb, candidate_emb, diff, mult], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 指南权威层级判定 =================
def get_guideline_tier(guideline_str):
    if not guideline_str: return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str: return 1
    elif "FIGO" in g_str: return 2
    elif "NCCN" in g_str: return 3
    else: return 4

TIER_NAMES = {
    1: "ESGO指南 (首选)", 2: "FIGO推荐 (次选)", 
    3: "NCCN指南 (推荐)", 4: "其他指南参考"
}

# ================= 获取图谱文献来源 =================
async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, 'driver'): return sources_info
            
        async with storage.driver.session() as session:
            # 兼容宏观聚合片段与单条语义片段
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
                
                sources_info.append({
                    "id": src_id, "pmid": final_pmid, 
                    "title": record.get("title") or "No Title", "guidelines": gl_str
                })
    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    return sources_info

# ================= PathoLLM：双语临床特征提取 =================
async def extract_bilingual_features(patient_case, client):
    sys_prompt = (
        "你是一个专门从事妇科肿瘤临床病理分析与文献检索特征提取的医学专家系统。\n"
        "你的任务是从用户输入的「病理报告」及「术后诊断/患者病史」中，精准提取出决定术后辅助治疗方案的核心临床特征，并将其转化为标准的中英文双语检索关键词。\n\n"
        "你必须严格按照以下步骤和规则进行提取，尤其要注意防错机制：\n\n"
        "1. 提取以下维度（若未提及当作“无”处理，绝对不可捏造）：疾病类型、组织学分级、肌层浸润深度、脉管内癌栓/LVSI状态、宫颈/子宫外受累情况、淋巴结转移情况、免疫组化与分子分型、FIGO分期、合并症与既往史。\n"
        "2. 必须将复合型分期拆解为独立的分期和分子特征。\n"
        "3. 患者的合并症可能会成为禁忌症，务必提炼为标准医学名词！\n"
        "4. 输出高密度的检索关键词，禁止输出完整的句子。\n\n"
        "必须严格按照以下格式输出，使用 <keywords> 标签包裹最终结果：\n"
        "<think>\n逐步分析过程...\n</think>\n"
        "<keywords>\n"
        "[中文检索词]: 子宫内膜样癌, Ⅰ级, 浸润浅肌层, p53野生型, FIGO IA期, 2型糖尿病\n"
        "[英文检索词]: Endometrial endometrioid adenocarcinoma, Grade 1 ...\n"
        "</keywords>"
    )
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": patient_case}],
            temperature=0.1, max_tokens=1500  
        )
        raw_content = response.choices[0].message.content.strip()
        match = re.search(r'<keywords>(.*?)</keywords>', raw_content, re.DOTALL | re.IGNORECASE)
        bilingual_keywords = match.group(1).strip() if match else re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
        print(f"  -> 成功提取双语特征:\n{bilingual_keywords}")
        return f"术后辅助治疗方案与临床指南 (Adjuvant treatment guidelines, recommendations and management)\n{bilingual_keywords}"
    except Exception as e:
        print(f"[Warning] 提取双语检索词失败: {e}")
        return patient_case

# ================= Qwen-Reranker 打分逻辑 (向量流拦截器) =================
async def compute_rerank_score(query, doc, client):
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
    except Exception: return 0.0

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
    print(f"\n>>> [引擎底层] 向量流触发独立 Rerank，正在为 {len(docs_list)} 条语义证据重新洗牌(Rank B)...")
    scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
    filtered_docs = [doc for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True) if score > 0.05]
    print(f"  └─ 洗牌完成，保留 {len(filtered_docs)}/{len(docs_list)} 条高质语义证据送入知识池。")
    return filtered_docs

async def main():
    print(">>> [加载模型] 正在挂载 ClinicalScoringNetwork 医疗裁判网络...")
    scorer_model = ClinicalScoringNetwork(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    try:
        scorer_model.load_state_dict(torch.load("clinical_scorer.pth"))
        scorer_model.eval()
        print("✅ 裁判网络加载成功！")
    except Exception as e:
        print(f"❌ 裁判网络加载失败: {e}")
        exit(1)

    print(f">>> [2/5] 正在初始化 PathoRAG 检索引擎...")
    try:
        graph_engine = PathoRAG(
            working_dir=WORKING_DIR, embedding_func=embedding_func,
            kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage", 
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"PathoRAG 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    patient_case = (
        """病理报告：\n"""
        """一、 全子宫： \n"""
        """1.子宫内膜样癌Ⅰ级，伴腺体粘液分化，癌灶大小1.2×1.2×0.4cm，浸润浅肌层（＜1/2肌层），\n"""
        """未见脉管内癌栓，向下未累及宫颈管；周围子宫内膜复杂不典型增生。 \n"""
        """2.慢性宫颈炎。 \n"""
        """二、 双侧卵巢包涵囊肿。双侧输卵管未见病变。 \n"""
        """三、（右侧前哨淋巴结（超分期））淋巴结（0/2）未见癌转移。 \n"""
        """（左侧前哨淋巴结（超分期））淋巴结（0/3）未见癌转移。 \n"""
        """免疫结果： \n"""
        """CK7（+），MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，80%，强），\n"""
        """PR（+，50%，强），P53（野生表型），Ki-67（10%），PTEN（+），β-catenin（膜+），\n"""
        """P16（斑驳+），L1CAM（-），AE1/AE3（-），AE1/AE3（-）。\n\n"""
        """术后诊断：\n"""
        """1.FIGO分期诊断：IA2期，2、2型糖尿病，3、气管支气管炎，4、慢性阻塞性肺炎，5、医疗个人史：腹腔镜检查史，右侧颈动脉支架放置术后"""
    )
    print(f"\n>>> [3/5] 输入原始病例:\n{patient_case}")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 执行双语临床特征提取...")
    enhanced_query = await extract_bilingual_features(patient_case, llm_client)
    print("\n>>> [4/5] 正在执行图谱混合检索与 神经网络 重排...")
    
    extended_knowledge_pool = {} 

    try:
        param = QueryParam(mode="hybrid", top_k=40, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(enhanced_query, param)
        
        if isinstance(retrieved_results, list):
            for item in retrieved_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    # 🚀 修复1：完美接住底层传上来的图谱信息熵推演分数
                    graph_score = float(item.get('<coherence>', 0.0))
                    
                    if content in ["RELATES_TO", "BELONG_TO", "EVIDENCE", ""] or len(content) < 5: 
                        continue
                        
                    # 🚀 修复2：尊重底层的宏观聚合，废弃业务层的多余图谱多跳检索
                    if content.startswith("【权威循证溯源："):
                        if "⚠️ 临床绝对警报" in content:
                            tag_info = "🚨 禁忌症熔断警报 (图谱绝对防守)"
                        else:
                            tag_info = "🧠 图谱高阶逻辑推演"
                        extended_knowledge_pool[content] = {"type": tag_info, "graph_score": graph_score}
                    else:
                        extended_knowledge_pool[content] = {"type": "🧩 纯向量语义召回", "graph_score": graph_score}

            # ==========================================
            # 🚀 深度医学逻辑融合法 (Graph Entropy + Neural Scoring)
            # ==========================================
            fragments_to_sort = []
            anchor_emb_np = await embedding_func(enhanced_query)
            anchor_tensor = torch.tensor(anchor_emb_np[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            candidate_contents = list(extended_knowledge_pool.keys())
            
            if candidate_contents:
                candidates_emb_np = await embedding_func(candidate_contents)
                candidates_tensor = torch.tensor(candidates_emb_np, dtype=torch.float32).to(DEVICE)
                anchor_tensor_expanded = anchor_tensor.expand(len(candidate_contents), -1)
                
                # 获取神经网络的纯语义打分
                with torch.no_grad():
                    neural_scores = scorer_model(anchor_tensor_expanded, candidates_tensor).cpu().numpy()
                
                for idx, content in enumerate(candidate_contents):
                    info = extended_knowledge_pool[content]
                    sources = await get_source_details(graph_engine, content)
                    best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
                    
                    raw_neural_score = float(neural_scores[idx])
                    graph_score = info["graph_score"]
                    
                    # ==========================================
                    # 🌟 核心算法：图谱拓扑与神经网络的加权融合
                    # ==========================================
                    if "🚨" in info["type"]:
                        # 触发禁忌症的文献，底层已经给 graph_score 加了极大分数
                        # 直接霸体保送，绝不让神经网络的低分把它刷掉！
                        final_score = graph_score + 1000.0  
                    elif "🧠" in info["type"]:
                        # 图谱逻辑推演出来的宏观方案：
                        # 用图谱分主导，神经网络分做微调
                        final_score = (graph_score * 0.7) + (raw_neural_score * 0.3)
                    else:
                        # 零散的纯向量召回片段：
                        # 此时完全相信双塔深度神经网络打分
                        final_score = raw_neural_score
                        
                    # 加上权威指南的微小加成
                    tier_bonus = (4 - best_tier) * 0.1 
                    final_score += tier_bonus
                    
                    fragments_to_sort.append({
                        "content": content, "info": info, "sources": sources,
                        "best_tier": best_tier, "neural_score": raw_neural_score, 
                        "graph_score": graph_score, "final_score": final_score
                    })
            
            # 按最终融合得分降序排列
            fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
            
            # 💡 精简输出：将最终喂给大模型的文献严格限制在 Top 10
            TOP_N_FOR_LLM = 10
            selected_fragments = fragments_to_sort[:TOP_N_FOR_LLM]

            print("\n" + "="*20 + f" Graph+Neural 融合重排顶级证据 (Top {len(selected_fragments)}) " + "="*20)
            bibliography, ref_counter, llm_context_list = {}, 1, []
            
            for idx, frag in enumerate(selected_fragments):
                source_indices = []
                for src in frag["sources"]:
                    src_id = src['id']
                    if src_id not in bibliography:
                        src['ref_index'] = ref_counter
                        bibliography[src_id] = src
                        ref_counter += 1
                    source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
                
                tier_name = TIER_NAMES[frag['best_tier']]
                ref_tag = f"【来源文献: {', '.join(source_indices)} | 证据级别: {tier_name}】" if source_indices else "【来源文献: 未知 | 证据级别: 缺乏指南支撑】"
                
                # 打印日志展示融合评分详情
                print(f"[{idx+1}] [{frag['info']['type']}] 复合终分: {frag['final_score']:.3f} (图谱推演分:{frag['graph_score']:.3f} | MLP预测分:{frag['neural_score']:.3f} | {tier_name})")
                print(f"内容: {frag['content'][:100]}...") 
                print("-" * 30)
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        return

    print("\n>>> [5/5] 正在生成治疗方案...")
    system_prompt = (
        "你是一个权威的妇科肿瘤临床辅助决策系统。\n"
        "请根据【参考证据】制定详细、可落地的术后辅助治疗方案。\n"
        "要求：\n"
        "1. 使用 <think>...</think> 进行循证推理分析。对于明确标注为【临床绝对警报/禁忌症】的证据，具有最高临床否决权，务必优先采纳并规避相关方案。\n"
        "2. 输出 <answer>...</answer> 结论。\n"
        "3. 方案必须极其详尽，必须明确随访频率、检查项目及内分泌/放化疗的周期和标准。\n"
        "4. 在建议中引用证据时，请直接使用原文中提供的序号标签，例如 '根据 [1] 的指南建议...'\n"
        "5. 【关键准则】：参考证据已根据底层信息熵打分排序，排名越靠前的方案优先级越高。"
    )
    user_prompt = f"【患者信息】\n{patient_case}\n\n【参考证据】\n{context_str}\n\n请制定极其详细的术后辅助治疗方案。"

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            stream=True, temperature=0.2, max_tokens=2048
        )
        print("\n>>> 模型回复:\n")
        async for chunk in response:
            if content := chunk.choices[0].delta.content: print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            for details in sorted(bibliography.values(), key=lambda x: x['ref_index']):
                idx, pmid_val, paper_id = details['ref_index'], details.get('pmid', 'Unknown'), details['id']
                print(f"[{idx}] PMID: {pmid_val}" if pmid_val != 'Unknown' else f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                print(f"    Title: {details['title']}\n    Guidelines: {details['guidelines']}\n" + "-" * 10)
        else: print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e: print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())