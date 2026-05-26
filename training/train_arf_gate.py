import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 1. 配置区域 =================
TRAIN_DATA_FILE = "/root/result/moe_training_data_llm_washed.json"  # 你的最新数据集
MODEL_SAVE_PATH = "/root/Model/moe_router_v2.pth"

# 文本向量化模型 (与你生成数据集时用的保持一致)
EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"

# 训练超参数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MARGIN = 0.4          # 边界损失的 Margin：正样本总分必须比负样本高出 0.1 以上
G_PENALTY_WEIGHT = 0.2 # 【核心】g值的惩罚权重，逼迫模型在安全时尽量使用纯语义(g趋于0)
COMPLEX_WEIGHT = 3.0  # 复杂病例的 Loss 权重放大倍数（对抗长尾分布）
# ===============================================

class MoEDataset(Dataset):
    def __init__(self, data_file):
        print(">>> 正在加载并解析数据集...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ 成功加载 {len(self.data)} 条训练数据。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "anchor_text": item["anchor_text"],
            "pos_semantic": torch.tensor(item["pos_semantic_score"], dtype=torch.float32),
            "neg_semantic": torch.tensor(item["neg_semantic_score"], dtype=torch.float32),
            "pos_graph": torch.tensor(item["pos_graph_score"], dtype=torch.float32),
            "neg_graph": torch.tensor(item["neg_graph_score"], dtype=torch.float32),
            "is_complex": torch.tensor(1.0 if item["is_complex"] else 1.0, dtype=torch.float32) 
            # 注意：返回权重倍数。如果是 complex，权重为 COMPLEX_WEIGHT，否则为 1.0
        }

class MoERouter(nn.Module):
    def __init__(self, input_dim, temperature_init=0.5):
        super(MoERouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)
        self.ln4 = nn.LayerNorm(64)
        self.fc5 = nn.Linear(64, 1)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.drop3(x)
        x = F.relu(self.ln4(self.fc4(x)))
        T = torch.exp(self.log_temperature)
        return torch.sigmoid(self.fc5(x) / T)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")

    # 1. 加载 Embedding 模型 (只用于提取特征，不参与训练，冻结梯度)
    print(">>> 正在加载 Embedding 模型...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    embedder.eval()
    embedder.to(device)
    embedding_dim = embedder.get_sentence_embedding_dimension()

    # 2. 准备数据
    dataset = MoEDataset(TRAIN_DATA_FILE)
    
    # 自定义 collate_fn 以处理字符串 batch
    def collate_fn(batch):
        anchor_texts = [item["anchor_text"] for item in batch]
        pos_semantic = torch.stack([item["pos_semantic"] for item in batch])
        neg_semantic = torch.stack([item["neg_semantic"] for item in batch])
        pos_graph = torch.stack([item["pos_graph"] for item in batch])
        neg_graph = torch.stack([item["neg_graph"] for item in batch])
        weights = torch.stack([item["is_complex"] for item in batch])
        
        # 将 complex flag 转换为权重：复杂为 COMPLEX_WEIGHT，普通为 1.0
        weights = torch.where(weights == 1.0, torch.tensor(COMPLEX_WEIGHT), torch.tensor(1.0))

        return anchor_texts, pos_semantic, neg_semantic, pos_graph, neg_graph, weights

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. 初始化 MoE 路由器和优化器
    model = MoERouter(input_dim=embedding_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\n>>> 开始训练 MoE 路由器...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_margin_loss = 0.0
        total_g = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            anchor_texts, pos_semantic, neg_semantic, pos_graph, neg_graph, weights = batch
            
            pos_semantic = pos_semantic.to(device).unsqueeze(1)
            neg_semantic = neg_semantic.to(device).unsqueeze(1)
            pos_graph = pos_graph.to(device).unsqueeze(1)
            neg_graph = neg_graph.to(device).unsqueeze(1)
            weights = weights.to(device).unsqueeze(1)

           # --- 前向传播：将 Anchor 转化为向量并计算 g ---
            with torch.no_grad():
                # 动态提取特征，不更新 embedder
                raw_embeddings = embedder.encode(anchor_texts, convert_to_tensor=True, normalize_embeddings=True)
                
            # 【关键修复】克隆并阻断梯度，将纯推理张量洗白为普通张量
            embeddings = raw_embeddings.clone().detach().to(device)
            # 为了确保后续能参与计算图（作为输入特征），明确要求计算梯度（可选，但最保险）
            embeddings.requires_grad_(True) 
                
            g = model(embeddings) # shape: (batch_size, 1)

            # --- 计算融合后的总分 ---
            # 最终得分 = g * 图谱分 + (1 - g) * 语义分
            pos_total_score = g * pos_graph + (1 - g) * pos_semantic
            neg_total_score = g * neg_graph + (1 - g) * neg_semantic

            # --- 自定义组合损失函数 (核心逻辑) ---
            # 1. 边界损失 (Margin Ranking Loss): 保证正样本得分大于负样本
            # 如果 neg_total_score 逼近甚至超过 pos_total_score，就会产生巨大的 Loss
            margin_loss_raw = F.relu(neg_total_score - pos_total_score + MARGIN)
            
            # 对复杂样本放大 Loss 惩罚，防止模型对长尾高危病例不敏感
            weighted_margin_loss = (margin_loss_raw * weights).mean()
            
            # 2. 门控极小化惩罚 (Gating Penalty)
            # 鼓励模型在能用语义区分（即 margin_loss == 0）时，尽量让 g 保持在较小的值。
            # 这能防止模型偷懒把所有的 g 都拉满到 1.0。
            g_penalty = g.mean() * G_PENALTY_WEIGHT

            # 3. 总 Loss
            loss = weighted_margin_loss + g_penalty

            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_margin_loss += weighted_margin_loss.item()
            total_g += g.mean().item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Avg g": f"{g.mean().item():.3f}"
            })

        avg_loss = total_loss / len(dataloader)
        avg_g = total_g / len(dataloader)
        print(f"📈 Epoch {epoch+1} 结束 | 平均总 Loss: {avg_loss:.4f} | 模型平均分配的 g 值: {avg_g:.3f}")

    # 保存训练好的 MoE 权重
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎉 训练圆满完成！MoE 路由器权重已保存至: {MODEL_SAVE_PATH}")
    print("下一步：您可以将此模型挂载到 test_mlp.py 业务管线中进行测试了！")

if __name__ == "__main__":
    train()