import os
import asyncio
from pathlib import Path
import numpy as np
import re
import math
from openai import AsyncOpenAI
from pathorag_core import PathoRAG, QueryParam
from pathorag_core.utils import wrap_embedding_func_with_attrs

# ================= 配置区域 =================

# LLM 服务配置 (此处使用你的 PathoLLM/OriClinical 模型)
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

# ===========================================
# 指南权威层级判定 (数值越小优先级越高)
def get_guideline_tier(guideline_str):
    if not guideline_str:
         return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str:
        return 1
    elif "FIGO" in g_str:
        return 2
    elif "NCCN" in g_str:
        return 3
    else:
        return 4

TIER_NAMES = {
    1: "ESGO指南 (首选)", 
    2: "FIGO推荐 (次选)", 
    3: "NCCN指南 (推荐)", 
    4: "其他指南参考"
}

REL_LEVEL_NAMES = {
    1: "强相关",
    2: "中等相关",
    3: "弱相关"
}

# ===========================================
# 获取图谱文献来源 (兼容单条超边与图谱聚合片段)
async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, 'driver'):
            return sources_info
            
        async with storage.driver.session() as session:
            # [新增分支] 识别图谱流的宏观聚合片段
            if knowledge_text.startswith("【权威循证溯源："):
                match = re.search(r'【权威循证溯源：(.*?)】', knowledge_text)
                if match:
                    paper_name = match.group(1).strip()
                    # 直接去查 Paper 节点
                    cypher_query = """
                    MATCH (paper:Paper)
                    WHERE paper.name = $paper_name OR paper.pmid = $paper_name
                    RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                    """
                    result = await session.run(cypher_query, paper_name=paper_name)
                    records = await result.data()
                else:
                    records = []
            else:
                # [原有分支] 识别向量流的单条超边
                node_id = f"<hyperedge>{knowledge_text}"
                cypher_query = """
                MATCH (target)-[r:BELONG_TO]->(paper:Paper)
                WHERE target.name = $node_id
                RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                """
                result = await session.run(cypher_query, node_id=node_id)
                records = await result.data()
            
            # 统一解析返回的记录
            for record in records:
                src_id = record.get("src_id", "Unknown").replace('"', '')
                raw_pmid = record.get("pmid")
                
                final_pmid = "Unknown"
                if raw_pmid and len(str(raw_pmid)) < 20: 
                    final_pmid = str(raw_pmid)
                elif "paper::" in src_id:
                    final_pmid = src_id.replace("paper::", "")
                    
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                
                sources_info.append({
                    "id": src_id,
                    "pmid": final_pmid, 
                    "title": record.get("title") or "No Title",
                    "guidelines": gl_str
                })
    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    
    return sources_info

# ===========================================
# PathoLLM：中英双语临床特征提取专家系统 (含病史与禁忌症提取)
async def extract_bilingual_features(patient_case, client):
    sys_prompt = (
        "你是一个专门从事妇科肿瘤临床病理分析与文献检索特征提取的医学专家系统。\n"
        "你的任务是从用户输入的「病理报告」及「术后诊断/患者病史」中，精准提取出决定术后辅助治疗方案的核心临床特征，并将其转化为标准的中英文双语检索关键词。\n\n"
        "你必须严格按照以下步骤和规则进行提取，尤其要注意我在“重点注意”中提出的防错机制：\n\n"
        "# 步骤1：核心临床特征分析与提取\n"
        "请仔细审查输入的所有信息，提取以下维度的特征（若未提及则忽略，绝对不可捏造）：\n"
        "1. 疾病类型 (Disease Type)\n"
        "2. 组织学分级 (Histological Grade)\n"
        "3. 肌层浸润深度 (Myometrial Invasion)\n"
        "4. 脉管内癌栓/LVSI状态 (LVSI Status)\n"
        "5. 宫颈/子宫外受累情况 (Extra-uterine Involvement)\n"
        "6. 淋巴结转移情况 (Lymph Node Metastasis)\n"
        "7. 免疫组化与分子分型 (Molecular Classification/IHC，如 MMR、p53 等)\n"
        "8. FIGO分期 (FIGO Stage)\n"
        "9. 合并症与既往史 (Comorbidities and Medical History，如糖尿病、呼吸系统疾病、心血管手术史等，这些对评估治疗禁忌症极其重要)\n\n"
        "## 重点注意！！！\n"
        "1. 当输入信息中没有提到相关特征时，当作“无”处理，绝对不可进行脑补或推断！！！\n"
        "2. 必须将复合型分期（如 IICmMMRd 或 IAmPOLEmut）拆解为独立的分期和分子特征。\n"
        "3. 患者的合并症（如2型糖尿病、慢阻肺、支架植入史等）可能会成为某些指南方案的【禁忌症】，请务必将其提炼为标准的医学名词一并输出！\n"
        "4. 你的目标是输出高密度的检索关键词，必须使用标准的临床医学术语，禁止输出完整的句子。\n\n"
        "# 步骤2：标准化双语输出\n"
        "请将步骤1中提取的有效特征，分别整理为中文和英文关键词列表（用逗号分隔）。\n"
        "必须严格按照以下格式输出，使用 <keywords> 标签包裹最终结果：\n"
        "<think>\n逐步分析过程...\n</think>\n"
        "<keywords>\n"
        "[中文检索词]: 子宫内膜样癌, Ⅰ级/低级别, 浸润浅肌层, p53野生型, FIGO IA期, 2型糖尿病, 慢性阻塞性肺炎, 颈动脉支架植入史...\n"
        "[英文检索词]: Endometrial endometrioid adenocarcinoma, Grade 1 / Low grade, Superficial myometrial invasion, p53 wild-type, FIGO stage IA, Type 2 diabetes mellitus, Chronic obstructive pulmonary disease (COPD), History of carotid artery stenting...\n"
        "</keywords>"
    )
    
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": patient_case}
            ],
            temperature=0.1,
            max_tokens=1500  
        )
        raw_content = response.choices[0].message.content.strip()
        
        match = re.search(r'<keywords>(.*?)</keywords>', raw_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            bilingual_keywords = match.group(1).strip()
        else:
            bilingual_keywords = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
        print(f"  -> 成功提取双语特征:\n{bilingual_keywords}")
        
        enhanced_query = (
            "术后辅助治疗方案与临床指南 (Adjuvant treatment guidelines, recommendations and management)\n"
            f"{bilingual_keywords}"
        )
        return enhanced_query
        
    except Exception as e:
        print(f"[Warning] 提取双语检索词失败: {e}")
        return patient_case

# ===========================================
# Qwen-Reranker 打分逻辑
async def compute_rerank_score(query, doc, client):
    instruction = "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan."
    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    query_text = f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
    
    raw_prompt = prefix + query_text + suffix

    try:
        response = await client.completions.create(
            model=RERANK_MODEL_NAME,
            prompt=raw_prompt,
            max_tokens=1,
            temperature=0,
            logprobs=20 
        )
        
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        
        true_logit = -10.0
        false_logit = -10.0
        
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes":
                true_logit = max(true_logit, logprob)
            elif clean_token == "no":
                false_logit = max(false_logit, logprob)
                
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        
        if true_score + false_score == 0:
            return 0.0
            
        score = true_score / (true_score + false_score)
        return score
    except Exception as e:
        print(f"[Rerank Warning] API打分失败: {e}")
        return 0.0


print(f">>> [1/5] 正在配置 API 客户端...")
try:
    embed_client = AsyncOpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)
    rerank_client = AsyncOpenAI(base_url=RERANK_API_URL, api_key=RERANK_API_KEY)
    
    @wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
    async def embedding_func(texts):
        if isinstance(texts, str): 
            texts = [texts]
        response = await embed_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=texts
        )
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings)
        
    print("API 客户端配置成功。")
except Exception as e:
    print(f"API 客户端配置失败: {e}")
    exit(1)


# ===========================================
# 专供底层引擎调用的 Rerank 包装器 (拦截向量流生成 Rank B)
# ===========================================
async def vector_stream_reranker(query_str, docs_list):
    if not docs_list: return []
    print(f"\n>>> [引擎底层] 向量流触发独立 Rerank，正在为 {len(docs_list)} 条语义证据重新洗牌(Rank B)...")
    
    scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
    
    scored_docs = sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True)
    filtered_docs = [doc for doc, score in scored_docs if score > 0.05]
    
    print(f"  └─ 洗牌完成，保留 {len(filtered_docs)}/{len(docs_list)} 条高质语义证据送入 W-RRF 融合池。")
    return filtered_docs


async def main():
    print(f">>> [2/5] 正在初始化 PathoRAG 检索引擎...")
    try:
        graph_engine = PathoRAG(
            working_dir=WORKING_DIR,
            embedding_func=embedding_func,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage", 
            graph_storage="Neo4JStorage", 
            reranker_func=vector_stream_reranker  # <--- 注入自定义的 Reranker 函数
        )
    except Exception as e:
        print(f"PathoRAG 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    # 替换为你最新的带有合并症和病史的病例输入
    patient_case = (
        """病理报告：\n"""
        """一、 全子宫： \n"""
        """1.子宫内膜样癌Ⅰ级，伴腺体粘液分化，癌灶大小1.2×1.2×0.4cm，浸润浅肌层（＜1/2肌层），\n"""
        """未见脉管内癌栓，向下未累及宫颈管；周围子宫内膜复杂不典型增生。 \n"""
        """2.慢性宫颈炎。 \n"""
        """二、 双侧卵巢包涵囊肿。 \n"""
        """双侧输卵管未见病变。 \n"""
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
    print(f"\n>>> 构建的高密度靶向检索词:\n{enhanced_query}")

    print("\n>>> [4/5] 正在执行图谱混合检索与 W-RRF 底层融合...")
    
    extended_knowledge_pool = {} 

    try:
        # 向引擎发起 Hybrid 查询，此时内部已经会自动调起 Reranker 洗牌并生成 W-RRF 分数
        param = QueryParam(mode="hybrid", top_k=40, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(enhanced_query, param)
        
        if isinstance(retrieved_results, list):
            for i, item in enumerate(retrieved_results):
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    w_rrf_score = float(item.get('<coherence>', 0.0))
                    
                    if content in ["RELATES_TO", "BELONG_TO", "EVIDENCE", ""] or len(content) < 5:
                        continue
                        
                    storage = graph_engine.chunk_entity_relation_graph
                    
                    # [核心判定]：区分为“图谱聚合片段”与“单条超边片段”
                    if content.startswith("【权威循证溯源："):
                        # 这是图谱流产出的结构化文献聚合方案，本身已经是黄金标准
                        if "⚠️ 临床绝对警报" in content:
                            tag_info = "图谱高定推演 (含禁忌症预警)"
                        else:
                            tag_info = "图谱逻辑推演 (多动作组合)"
                        extended_knowledge_pool[content] = {"type": tag_info, "w_rrf_score": w_rrf_score}
                        
                    else:
                        # 这是向量流产出的单条超边片段，走验证逻辑
                        node_id = f"<hyperedge>{content}"
                        if hasattr(storage, 'driver'):
                            async with storage.driver.session() as session:
                                check_query = """
                                MATCH (e)-[r:RELATES_TO]->(h)
                                WHERE h.name = $node_id
                                RETURN DISTINCT r.role AS role
                                """
                                check_res = await session.run(check_query, node_id=node_id)
                                role_recs = await check_res.data()
                                roles = [rec["role"] for rec in role_recs if rec["role"]]
                                
                                has_core_role = "RECOMMENDATION" in roles or "CONTRAINDICATION" in roles
                                
                                if has_core_role:
                                    tag_info = f"直接命中 (包含: {','.join(roles)})"
                                    extended_knowledge_pool[content] = {"type": tag_info, "w_rrf_score": w_rrf_score}
                                else:
                                    tag_info = f"背景知识 (包含: {','.join(roles) if roles else '未知'})"
                                    extended_knowledge_pool[content] = {"type": tag_info, "w_rrf_score": w_rrf_score}
                                    
                                    expand_query = """
                                    MATCH (start_h)<-[:RELATES_TO]-(e)-[r2:RELATES_TO]->(target_h)
                                    WHERE start_h.name = $node_id 
                                      AND target_h.name STARTS WITH '<hyperedge>'
                                      AND start_h <> target_h
                                      AND r2.role IN ['RECOMMENDATION', 'CONTRAINDICATION']
                                    RETURN DISTINCT target_h.name AS neighbor_id, r2.role AS target_role
                                    LIMIT 3
                                    """
                                    expand_res = await session.run(expand_query, node_id=node_id)
                                    expanded_recs = await expand_res.data()
                                    
                                    for rec in expanded_recs:
                                        neighbor_content = rec["neighbor_id"].replace("<hyperedge>", "")
                                        target_role = rec["target_role"]
                                        if neighbor_content not in extended_knowledge_pool:
                                            # 多跳补充证据：继承主节点的 W-RRF 分数并做一定的折损衰减
                                            extended_knowledge_pool[neighbor_content] = {
                                                "type": f"多跳补充 ({target_role})", 
                                                "w_rrf_score": w_rrf_score * 0.8
                                            }
                        else:
                            extended_knowledge_pool[content] = {"type": "直接命中 (非Neo4j)", "w_rrf_score": w_rrf_score}

            # ==========================================
            # 临床分档融合法 (Stratified Ranking)
            # 基于底层提供的 W-RRF 分数进行分档
            # ==========================================
            fragments_to_sort = []
            
            for content, info in extended_knowledge_pool.items():
                sources = await get_source_details(graph_engine, content)
                
                best_tier = 4 # 默认最低优先级
                for src in sources:
                    current_tier = get_guideline_tier(src.get('guidelines', ''))
                    if current_tier < best_tier:
                        best_tier = current_tier
                
                w_rrf_score = info.get("w_rrf_score", 0.0)
                
                # 重新划定 W-RRF 综合相关度档位
                if w_rrf_score >= 0.03:
                    rel_level = 1  # 强相关：双路头部共同命中或单路绝对高分
                elif w_rrf_score >= 0.015:
                    rel_level = 2  # 中相关：中段命中
                else:
                    rel_level = 3  # 弱相关：长尾补充
                
                fragments_to_sort.append({
                    "content": content,
                    "info": info,
                    "sources": sources,
                    "best_tier": best_tier,
                    "w_rrf_score": w_rrf_score,
                    "rel_level": rel_level
                })
            
            # 三维复合排序：相关度档位(升序) -> 指南权威度(升序) -> W-RRF绝对分数(降序)
            fragments_to_sort.sort(key=lambda x: (x["rel_level"], x["best_tier"], -x["w_rrf_score"]))
            
            # 截断：只取排名前 15 的最优知识片段喂给 LLM
            TOP_N_FOR_LLM = 15
            selected_fragments = fragments_to_sort[:TOP_N_FOR_LLM]

            # ==========================================
            # 构建参考文献字典与组装 LLM 上下文
            # ==========================================
            print("\n" + "="*20 + f" 最终筛选证据片段 (Top {len(selected_fragments)}) " + "="*20)
            
            bibliography = {}
            ref_counter = 1 
            llm_context_list = []
            
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
                rel_name = REL_LEVEL_NAMES[frag['rel_level']]
                
                if source_indices:
                    ref_tag = f"【来源文献: {', '.join(source_indices)} | 证据级别: {tier_name}】"
                else:
                    ref_tag = f"【来源文献: 未知 | 证据级别: 缺乏指南支撑】"
                
                display_src = ", ".join(source_indices) if source_indices else "[未知]"
                # 打印日志展示分档排序详情，此时打印的分数已变为 W-RRF
                print(f"片段 [{idx+1}] [{frag['info']['type']}] 来源: {display_src} ")
                print(f"  --> [评估]: {rel_name}(分档:{frag['rel_level']}) | {tier_name}(层级:{frag['best_tier']}) | W-RRF得分: {frag['w_rrf_score']:.4f}")
                print(f"内容: {frag['content'][:80]}...") 
                print("-" * 30)
                
                # 拼接给 LLM 的最终上下文
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n>>> [5/5] 正在生成治疗方案...")
    
    system_prompt = (
        "你是一个权威的妇科肿瘤临床辅助决策系统。\n"
        "请根据【参考证据】制定详细、可落地的术后辅助治疗方案。\n"
        "要求：\n"
        "1. 使用 <think>...</think> 进行循证推理分析。\n"
        "2. 输出 <answer>...</answer> 结论。\n"
        "3. 方案必须极其详尽，禁止使用“定期随访”、“酌情化疗”等模糊表述。必须明确随访频率、检查项目及内分泌/放化疗的周期和标准。\n"
        "4. 在建议中引用证据时，请直接使用原文中提供的序号标签，例如 '根据 [1] 的指南建议...'\n"
        "5. 【关键准则】：参考证据已标注“证据级别”。你的推荐权重必须倾向于“首选”级别的指南，并在说明中体现采纳的优先级。"
    )

    user_prompt = f"""【患者信息】
{patient_case}

【参考证据】
{context_str}

请制定极其详细的术后辅助治疗方案。"""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.2,
            max_tokens=2048
        )

        print("\n>>> 模型回复:\n")
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            sorted_refs = sorted(bibliography.values(), key=lambda x: x['ref_index'])
            
            for details in sorted_refs:
                idx = details['ref_index']
                pmid_val = details.get('pmid', 'Unknown')
                paper_id = details['id']
                
                if pmid_val != 'Unknown':
                    print(f"[{idx}] PMID: {pmid_val}")
                else:
                    print(f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                    
                print(f"    Title: {details['title']}")
                print(f"    Guidelines: {details['guidelines']}")
                print("-" * 10)
        else:
            print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e:
        print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())