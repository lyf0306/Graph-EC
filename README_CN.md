# PathoRAG — 面向子宫内膜癌的两阶段临床决策支持系统

一个将**微调 LLM 分期**（PathoLLM）与**超图增强检索**（PathoRAG）相结合的临床决策支持系统（CDSS），用于子宫内膜癌的循证治疗推荐。

## 系统架构

```
阶段一：PathoLLM                       阶段二：PathoRAG
（微调 FIGO 分期）                      （KG + HGNN + MoE CDSS）

病理报告                               患者病历 + FIGO 分期
     │                                        │
     ▼                                        ▼
DeepSeek-R1-32B + LoRA              KG 检索 ←→ 向量检索
     │                                        │
     ▼                                        ▼
FIGO 2023 分期 + 推理链              MoE 动态融合（ARF 门控）
                                              │
                                              ▼
                                         HGNN 证据评分
                                              │
                                              ▼
                                         合并症 Agent 筛查
                                              │
                                              ▼
                                         MDT 报告（ESGO 风险 + 治疗推荐）
```

## 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| **PathoRAG 引擎** | `pathorag_core/pathorag.py` | 混合图+向量检索，带一致性评分 |
| **HGNN 评分器** | `pathorag_core/hyper_attention.py` | 多头超图注意力机制，用于临床证据排序 |
| **ARF 门控** | `training/train_arf_gate.py` | 3层 MLP（960→256→64→1）图/向量双流动态融合 |
| **CAES** | `training/train_caes.py` | 交叉注意力实体评分器，基于成对排序损失 |
| **实体抽取** | `pathorag_core/operate.py` | 从临床文本中提取 5 种语义角色实体 |
| **REST API** | `api/` | FastAPI 服务，9 阶段 MDT 管线 |
| **Agent** | `agent/` | 合并症筛查与治疗安全性评估 |

## 数据管线

```
原始患者病历（Word/PDF）
    │  data/extract_patients.py：LLM API 提取结构化特征
    ▼
clinical_dataset_v3.jsonl
    │  data/build_training_data.py：三阶段神经符号精炼
    │    (i)  关键词预筛选
    │    (ii) LLM 评委质控（temperature=0.0）
    │    (iii) 图评分阈值（pos_graph ≥ 0.1）
    ▼
moe_training_data_v2_subgraph.json（ARF 门控 + CAES 训练数据）
    │
    ├──► training/train_arf_gate.py：训练 ARF 门控（BCE 损失，实值图评分）
    └──► training/train_caes.py：训练 CAES（成对排序损失）
```

### 负样本挖掘（2 条 KG 规则）

1. **FIGO 分期错配**——分期错误 → 治疗方案错误
2. **禁忌症诱导**——仅对高危合并症的复杂病例触发

### 神经符号图评分

PathoRAG 混合推理（`mode="hybrid"`，top_k=50）输出的实值评分：

```
score = coherence × sim^0.5
```

其中 `sim` 为重排序语义相似度（阈值 > 0.4，最低 0.01）。

## 快速开始

### 环境要求

- Python 3.10+
- Neo4j（图数据库）
- Milvus（向量数据库）
- Embedding 服务（兼容 OpenAI 接口，如 QwenEmbedding + vLLM）
- Reranker 服务（兼容 OpenAI 接口，如 QwenReranker + vLLM）
- LLM API 密钥（兼容 OpenAI 接口：DeepSeek / Qwen / GPT 等）

### 安装配置

```bash
cd PathoRAG

# 安装依赖
pip install -r requirements-web.txt
pip install -r requirements-observability.txt

# 核心依赖
pip install neo4j pymilvus sentence-transformers openai torch

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API 密钥和数据库地址

# 验证安装
python -c "from pathorag_core import PathoRAG, QueryParam; print('OK')"
```

### 启动 API 服务

```bash
cd api
python main.py
# → http://localhost:8000
# → /health — 健康检查
# → /analyze — 完整 MDT 管线
# → /docs — Swagger 文档
```

## 关键文件

### 训练数据构建
- `data/extract_patients.py` — 从原始病历中提取结构化患者特征
- `data/build_training_data.py` — ARF 门控 + CAES 数据的神经符号精炼管线
- `data/build_moe_trainset.py` — MoE 训练集构建
- `data/compute_idf.py` — 实体-角色 IDF 计算

### 模型训练
- `training/train_arf_gate.py` — ARF 门控（MoE 路由器）训练
- `training/train_caes.py` — CAES（交叉注意力证据评分器）训练
- `training/train_scorer.py` — 证据评分器训练

### 测试
- `tests/test_pipeline.py` — 主集成测试（全管线）
- `tests/test_workflow.py` — 工作流验证
- `tests/test_mlp.py` — MLP 组件单元测试

### API
- `api/main.py` — FastAPI 应用入口
- `api/config.py` — 配置管理（基于 .env 的 pydantic-settings）
- `api/services/pipeline_service.py` — 9 阶段 MDT 管线编排
- `api/services/resource_manager.py` — PathoRAG + CAES 资源生命周期管理
- `api/services/patient_retriever.py` — KNN + XGBoost 患者相似性检索

## ESGO 2025 风险分级

PathoRAG 实现了完整的 ESGO 2025 决策树：
- 低危 / 中危 / 高-中危 / 高危 / 晚期 / 转移性
- 整合：FIGO 分期、组织学类型、分级、LVSI、肌层浸润、分子分型

## MDT 报告（9 阶段管线）

1.  患者画像提取（LLM）
2.  结构化特征提取（40+ 字段）
3.  ESGO 2025 风险分级
4.  知识图谱查询（图检索 + 向量检索）
5.  MoE 动态权重计算
6.  KG 证据检索 + HGNN 排序
7.  重排序 + 参考文献格式化
8.  相似患者检索 + 治疗预测（KNN + XGBoost）
9.  合并症筛查 Agent → 最终 MDT 报告

## 可观测性

可选启用 OpenTelemetry + Prometheus 监控：

```python
from pathorag_core.instrumentation import init_instrumentation
init_instrumentation(service_name="pathorag")
```

导出指标：
- PathoRAG 查询延迟直方图
- ARF 门控融合权重 α 分布
- CAES 注意力分数分布
- 管线各阶段耗时追踪

## 许可证

专有软件——仅限研究使用。
