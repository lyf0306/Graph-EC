import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_PATH = "/root/result/contrastive_training_dataset_final.jsonl"
MODEL_SAVE_PATH = "clinical_scorer.pth"
EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"
# ===========================================

# 1. 定义对比学习打分网络
class ClinicalScoringNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(ClinicalScoringNetwork, self).__init__()
        # 融合特征维度：Anchor, Candidate, 绝对差值, 逐元素相乘
        self.fc1 = nn.Linear(embedding_dim * 4, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(128, 1) # 输出标量打分

    def forward(self, anchor_emb, candidate_emb):
        diff = torch.abs(anchor_emb - candidate_emb)
        mult = anchor_emb * candidate_emb
        # 拼接: [u, v, |u-v|, u*v] 
        x = torch.cat([anchor_emb, candidate_emb, diff, mult], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        score = self.out(x)
        return score.squeeze(-1) 

# 2. 极速缓存版数据集
class FastTumorBoardDataset(Dataset):
    def __init__(self, jsonl_file, embed_model):
        self.data = []
        unique_texts = set()
        
        print(">>> [1/3] 正在解析 JSONL 数据...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
                    unique_texts.add(item["anchor"])
                    unique_texts.add(item["positive"])
                    for neg in item["hard_negatives"]:
                        unique_texts.add(neg)
                        
        print(f">>> [2/3] 共有 {len(unique_texts)} 条独立文本，正在进行离线 Embedding 预计算...")
        unique_texts = list(unique_texts)
        # 一次性算出所有向量并存在字典里，后续训练速度起飞
        embeddings = embed_model.encode(unique_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        self.emb_dict = {text: emb for text, emb in zip(unique_texts, embeddings)}
        print("✅ 离线特征提取完成！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor_vec = self.emb_dict[item["anchor"]]
        pos_vec = self.emb_dict[item["positive"]]
        
        # 获取当前所有的负样本向量
        neg_vecs = [self.emb_dict[neg] for neg in item["hard_negatives"]]
        
        # ⚠️ 核心修复：强制对齐负样本数量到固定维度 (例如 3 个)
        # 如果少于 3 个，就循环重复已有的负样本进行 Padding
        while len(neg_vecs) < 3:
            neg_vecs.append(neg_vecs[-1]) 
            
        # 如果多于 3 个，强行截断，保证维度绝对一致
        neg_vecs = neg_vecs[:3]
            
        return {
            "anchor": torch.tensor(anchor_vec, dtype=torch.float32),
            "positive": torch.tensor(pos_vec, dtype=torch.float32),
            "negatives": torch.stack([torch.tensor(n, dtype=torch.float32) for n in neg_vecs])
        }

# 3. InfoNCE Loss
def info_nce_loss(pos_scores, neg_scores, temperature=0.15):
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    logits = logits / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# 4. 训练主循环
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    EMB_DIM = embed_model.get_sentence_embedding_dimension()
    print(f"检测到 Embedding 维度为: {EMB_DIM}")
    
    dataset = FastTumorBoardDataset(DATASET_PATH, embed_model)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ClinicalScoringNetwork(embedding_dim=EMB_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    
    epochs = 10
    print("\n>>> [3/3] 开始对比学习打分网络训练...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            anchor = batch["anchor"].to(device)       
            positive = batch["positive"].to(device)   
            negatives = batch["negatives"].to(device) 
            
            optimizer.zero_grad()
            
            pos_scores = model(anchor, positive)      
            
            batch_size, num_negs, dim = negatives.shape
            anchor_expanded = anchor.unsqueeze(1).expand(-1, num_negs, -1).reshape(-1, dim)
            negatives_flat = negatives.reshape(-1, dim)
            
            neg_scores_flat = model(anchor_expanded, negatives_flat)
            neg_scores = neg_scores_flat.reshape(batch_size, num_negs) 
            
            loss = info_nce_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ 模型训练完毕，权重已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()