import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 1. 全局配置区域 =================
TRAIN_DATA_FILE = "/root/result/moe_training_data_v2_subgraph.json" 
MODEL_SAVE_PATH = "/root/Model/clinical_attention_v3.pth" # 升级为 V3 HGNN 版

EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"
BATCH_SIZE = 32
EPOCHS = 40           
LEARNING_RATE = 1e-4  # 引入真正的自注意力聚合，调低学习率求稳
MARGIN = 0.4          

NUM_HEADS = 8         
HEAD_DIM = 128        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROLE_VOCAB = {"EVIDENCE": 0, "CONTEXT": 1, "CONDITION": 2, "RECOMMENDATION": 3, "CONTRAINDICATION": 4, "PAD": 5}
MAX_ENTITIES = 40 
# ===============================================

class TrueHypergraphDataset(Dataset):
    def __init__(self, data_file, embedding_model):
        print(">>> 正在解析超图数据集...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
            
        self.queries = [item["anchor_text"] for item in self.raw_data]
        self.emb_dim = embedding_model.get_sentence_embedding_dimension()
        
        unique_entities = set()
        for item in self.raw_data:
            for ent in item.get("positive_subgraph", []): unique_entities.add(ent["entity"])
            for ent in item.get("negative_subgraph", []): unique_entities.add(ent["entity"])
        unique_entities = list(unique_entities)
        
        print(f">>> 预计算 {len(unique_entities)} 个实体的 Embedding (请稍候)...")
        with torch.no_grad():
            self.query_embs = embedding_model.encode(self.queries, convert_to_tensor=True).cpu()
            if len(unique_entities) > 0:
                ent_embs = embedding_model.encode(unique_entities, convert_to_tensor=True).cpu()
                self.ent_emb_map = {ent: ent_embs[i] for i, ent in enumerate(unique_entities)}
            else:
                self.ent_emb_map = {}

        print(">>> 正在组装超图拓扑张量 (彻底废弃人工 IDF)...")
        self.pos_data, self.neg_data = [], []
        for item in self.raw_data:
            self.pos_data.append(self._pad_subgraph(item.get("positive_subgraph", [])))
            self.neg_data.append(self._pad_subgraph(item.get("negative_subgraph", [])))

    def _pad_subgraph(self, subgraph):
        embs, roles, mask = [], [], []
        for ent in subgraph[:MAX_ENTITIES]:
            embs.append(self.ent_emb_map[ent["entity"]])
            roles.append(ROLE_VOCAB.get(ent["role"].upper(), 1)) 
            mask.append(1.0) # 1.0 表示有效节点
            
        pad_len = MAX_ENTITIES - len(embs)
        if pad_len > 0:
            embs.extend([torch.zeros(self.emb_dim)] * pad_len)
            roles.extend([ROLE_VOCAB["PAD"]] * pad_len)
            mask.extend([0.0] * pad_len) # 0.0 表示 PAD 无效节点
            
        # 🌟 删除了返回 IDF 的逻辑
        return (torch.stack(embs), torch.tensor(roles, dtype=torch.long), torch.tensor(mask, dtype=torch.bool))

    def __len__(self): return len(self.raw_data)
    def __getitem__(self, idx): return self.query_embs[idx], self.pos_data[idx], self.neg_data[idx]

# ================= 3. 破茧重生：真正的超图神经网络 (HGNN) =================
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
        # 深度交叉评分网络 (注意：输入维度从 4*head_dim 变成了 5*head_dim)
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
        
        # 强制走真正的图学习前向传播（彻底废弃散装实体兼容）
        ent_embs, roles, mask = target_data
        max_ent = ent_embs.size(1)
        
        # Step 1: 节点特征融合 (实体文本特征 + 极度缩放的角色特征)
        r_embs = self.role_embeddings(roles)
        # 核心修复 3：逼迫网络主要依赖 ent_embs，角色只是一个微弱的 Bias
        H_0 = self.node_norm(ent_embs + r_embs * self.role_scale) # [Batch, MaxEnt, EmbDim]
        
        # Step 2: 真正的超图消息传递 (Message Passing)
        key_padding_mask = (mask == 0.0) 
        msg_out, _ = self.msg_passing(H_0, H_0, H_0, key_padding_mask=key_padding_mask)
        H_1 = self.post_msg_norm(H_0 + msg_out) # 此时 H_1 包含了超边内所有实体的拓扑上下文
        
        # Step 3: Query 与节点表达的深度交叉
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        
        # 核心修复 4：保留原始纯文本的 K，防止拓扑融合后特征坍缩
        k_msg = self.dropout(F.gelu(self.W_k(H_1)))       # 含有拓扑和角色信息的 K
        k_raw = self.dropout(F.gelu(self.W_k(ent_embs)))  # 纯净的原始文本 K 
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k_msg = k_msg.view(batch_size, max_ent, self.num_heads, self.head_dim)
        k_raw = k_raw.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k_msg) 
        
        # 核心修复 5：将原始文本差异与拓扑差异合并，逼迫 MLP 必须看文本特征！
        interaction = torch.cat([
            q_expanded, 
            k_msg, 
            torch.abs(q_expanded - k_msg),  # 拓扑语义差距
            torch.abs(q_expanded - k_raw),  # 纯文本语义差距 (强制引入)
            q_expanded * k_msg
        ], dim=-1)
        
        cross_scores = self.interaction_scorer(interaction).squeeze(-1) # [Batch, MaxEnt]
        
        # Step 4: 屏蔽无效节点并进行均值池化 (Mean Pooling) 替代 Sum
        mask_expanded = mask.unsqueeze(-1).float()
        valid_node_scores = cross_scores * mask_expanded
        
        # 核心修复 6：按实际有效节点数取平均，消除长超边偏见
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0) 
        hyperedge_scores = valid_node_scores.sum(dim=1) / valid_counts
        
        final_score = hyperedge_scores.mean(dim=-1)
        weights = torch.sigmoid(final_score)
        
        if self.training:
            return weights, q.squeeze(1)
        return weights

def orthogonal_loss(q_heads):
    q_norm = F.normalize(q_heads, p=2, dim=-1)
    sim_matrix = torch.bmm(q_norm, q_norm.transpose(1, 2)) 
    identity = torch.eye(q_heads.size(1), device=q_heads.device).unsqueeze(0)
    return (sim_matrix - identity).pow(2).sum(dim=(1, 2)).mean()

def train():
    print(f">>> 当前计算设备: {DEVICE}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    emb_dim = embedding_model.get_sentence_embedding_dimension()
    
    dataset = TrueHypergraphDataset(TRAIN_DATA_FILE, embedding_model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del embedding_model 
    torch.cuda.empty_cache()

    model = EndToEndHypergraphNetwork(embedding_dim=emb_dim, num_heads=NUM_HEADS, head_dim=HEAD_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MarginRankingLoss(margin=MARGIN)

    best_loss = float('inf')
    ORTHO_WEIGHT = 0.1 
    
    print(">>> 🚀 开始训练 [纯正学术版] 端到端超图神经网络...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_pos, total_neg = 0.0, 0.0, 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for q_emb, pos_data, neg_data in progress_bar:
            q_emb = q_emb.to(DEVICE)
            pos_data = [d.to(DEVICE) for d in pos_data]
            neg_data = [d.to(DEVICE) for d in neg_data]
            
            optimizer.zero_grad()
            pos_score, q_heads_pos = model(q_emb, pos_data)
            neg_score, _ = model(q_emb, neg_data)
            
            y = torch.ones_like(pos_score).to(DEVICE)
            loss_rank = criterion(pos_score, neg_score, y)
            loss_ortho = orthogonal_loss(q_heads_pos)
            
            loss = loss_rank + ORTHO_WEIGHT * loss_ortho
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pos += pos_score.mean().item()
            total_neg += neg_score.mean().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "PosW": f"{pos_score.mean().item():.2f}", "NegW": f"{neg_score.mean().item():.2f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Avg Pos: {total_pos/len(dataloader):.3f} | Avg Neg: {total_neg/len(dataloader):.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()