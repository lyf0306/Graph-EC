# pathorag_core/hyper_attention.py
import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🌟 1. 放入纯正学术版：端到端超图神经网络 (HGNN)
class EndToEndHypergraphNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # ================= 核心修复 1 =================
        # 缩小角色的初始化权重，防止起步阶段掩盖文本特征
        self.role_embeddings = nn.Embedding(6, embedding_dim)
        nn.init.normal_(self.role_embeddings.weight, std=0.01) # 强制极小初始化
        
        # 引入一个可学习的缩放标量，初始给予极小影响
        self.role_scale = nn.Parameter(torch.tensor(0.05)) 
        # ============================================

        self.node_norm = nn.LayerNorm(embedding_dim)
        
        # 真正的图消息传递层 (Message Passing)
        self.msg_passing = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        self.post_msg_norm = nn.LayerNorm(embedding_dim)
        
        self.W_q = nn.Linear(embedding_dim, num_heads * head_dim)
        self.W_k = nn.Linear(embedding_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(0.2)
        
        # ================= 核心修复 2 =================
        # 深度交叉评分网络 (输入维度: 5*head_dim)
        self.interaction_scorer = nn.Sequential(
            nn.Linear(5 * head_dim, 2 * head_dim),
            nn.GELU(),
            nn.LayerNorm(2 * head_dim),
            nn.Dropout(0.15),
            
            nn.Linear(2 * head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            
            nn.Linear(head_dim, 1) 
        )

    def forward(self, q_emb, target_data):
        batch_size = q_emb.size(0)
        
        ent_embs, roles, mask = target_data
        max_ent = ent_embs.size(1)
        
        # Step 1: 节点特征融合 (实体文本特征 + 极度缩放的角色特征)
        r_embs = self.role_embeddings(roles)
        H_0 = self.node_norm(ent_embs + r_embs * self.role_scale) 
        
        # Step 2: 真正的超图消息传递 (Message Passing)
        key_padding_mask = (mask == 0.0) 
        msg_out, _ = self.msg_passing(H_0, H_0, H_0, key_padding_mask=key_padding_mask)
        H_1 = self.post_msg_norm(H_0 + msg_out) 
        
        # Step 3: Query 与节点表达的深度交叉
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        
        k_msg = self.dropout(F.gelu(self.W_k(H_1)))       
        k_raw = self.dropout(F.gelu(self.W_k(ent_embs)))  
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k_msg = k_msg.view(batch_size, max_ent, self.num_heads, self.head_dim)
        k_raw = k_raw.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k_msg) 
        
        interaction = torch.cat([
            q_expanded, 
            k_msg, 
            torch.abs(q_expanded - k_msg),  
            torch.abs(q_expanded - k_raw),  
            q_expanded * k_msg
        ], dim=-1)
        
        cross_scores = self.interaction_scorer(interaction).squeeze(-1) 
        
        # Step 4: 屏蔽无效节点并进行均值池化
        mask_expanded = mask.unsqueeze(-1).float()
        valid_node_scores = cross_scores * mask_expanded
        
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0) 
        hyperedge_scores = valid_node_scores.sum(dim=1) / valid_counts
        
        final_score = hyperedge_scores.mean(dim=-1)
        weights = torch.sigmoid(final_score)
        
        if self.training:
            return weights, q.squeeze(1)
        return weights

# 2. 全局单例与缓存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTENTION_MODEL = None

# 【非常重要】初始化为空字典，不能被重新赋值 '=' 覆盖
GLOBAL_ENTITY_CACHE = {}

def init_attention_system(model_path: str, vdb_path: str, embedding_dim: int):
    global ATTENTION_MODEL, GLOBAL_ENTITY_CACHE
    if ATTENTION_MODEL is not None: return 
        
    print(">>> 🧠 正在唤醒 End-to-End Hypergraph Neural Network (HGNN) 临床直觉模块...")
    
    # ========================================================
    # 彻底抛弃旧的 JSON 库，直连 Milvus 全量拉取 2560 维特征
    # ========================================================
    try:
        from pymilvus import MilvusClient
        milvus_uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
        client = MilvusClient(uri=milvus_uri)
        
        # 自动探测实体集合名称 (通常包含 entity 或 entities)
        collections = client.list_collections()
        target_coll = next((c for c in collections if "entity" in c.lower() or "entities" in c.lower()), None)
        
        if not target_coll:
            print("  ❌ 警告: 在 Milvus 中没有找到名字包含 'entity' 的集合！无法加载缓存。")
        else:
            print(f"  -> 正在从 Milvus 集合 [{target_coll}] 拉取全量 2560 维实体向量...")
            
            # 使用迭代器防止单次 query 超出 limit 限制
            iterator = client.query_iterator(
                collection_name=target_coll, 
                output_fields=["id", "entity_name", "vector"], 
                batch_size=5000
            )
            
            temp_cache = {}
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for item in batch:
                    # 自适应字段名：提取实体名和向量
                    ent_name = item.get("entity_name") or item.get("name") or str(item.get("id"))
                    vec = item.get("vector") or item.get("embedding")
                    if ent_name and vec is not None:
                        # 存为 CPU 上的 Tensor
                        temp_cache[str(ent_name).upper()] = torch.tensor(vec, dtype=torch.float32)
            
            # 【核心修复】：原地更新全局字典，保持外层引用指针不丢失
            GLOBAL_ENTITY_CACHE.clear()
            GLOBAL_ENTITY_CACHE.update(temp_cache)
            
            print(f"  ✔ 成功加载了 {len(GLOBAL_ENTITY_CACHE)} 个实体的 2560 维特征！")
            
    except Exception as e:
        print(f"  ❌ 从 Milvus 拉取实体向量失败: {e}")
        print(f"  ⚠️ 请检查 os.environ['MILVUS_URI'] 是否配置正确。")

    # 🌟 实例化新版 HGNN 模型
    ATTENTION_MODEL = EndToEndHypergraphNetwork(
        embedding_dim=embedding_dim, 
        num_heads=8, 
        head_dim=128
    )
    
    ATTENTION_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ATTENTION_MODEL.to(DEVICE)
    ATTENTION_MODEL.eval() 
    print(">>> ✔ 纯正学术版 (HGNN) 临床直觉网络初始化完成并已进入推理模式！")

def compute_hyperedge_scores_sync(query_tensor, hyperedges_data):
    """
    同步计算超边的整体推荐度
    hyperedges_data: (ent_embs_batch, roles_batch, mask_batch)
    """
    with torch.no_grad():
        ent_embs_batch, roles_batch, mask_batch = hyperedges_data
        
        batch_size = ent_embs_batch.size(0)
        q_emb_batch = query_tensor.unsqueeze(0).expand(batch_size, -1).to(DEVICE)
        
        ent_embs_batch = ent_embs_batch.to(DEVICE)
        roles_batch = roles_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        
        weights = ATTENTION_MODEL(q_emb_batch, (ent_embs_batch, roles_batch, mask_batch))
        
        return weights.cpu().numpy()