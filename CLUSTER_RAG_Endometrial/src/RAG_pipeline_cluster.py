"""
RAG_pipeline_v3.py
适配新版 XGBoost + 软聚类模型（v3），使用本地 vLLM 进行生成
"""
import json
import numpy as np
import pandas as pd
import joblib
import openai  # 替换 requests，用于 vLLM
import sys  # 新增
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 新增：导入格式化工具函数
from utils.trans_format import format_patient_desc

# ===================== vLLM 配置（本地部署） =====================
VLLM_API_KEY = "EMPTY"
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL_NAME = "OriClinical"

client = openai.OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

def call_vllm(prompt, temperature=0.1):
    """调用本地 vLLM 模型生成回答"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=6000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"vLLM API 调用失败: {e}"

# ===================== 路径 =====================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "structured_output_v2.json"          # 使用新版数据
MODEL_PATH = BASE_DIR / "models" / "trained_xgb_softcluster_v3.pkl" # 模型字典
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor_v3.pkl"      # 基础预处理器
KMEANS_PATH = BASE_DIR / "models" / "kmeans_v3.pkl"                  # KMeans 聚类模型

# 预测阈值（固定）
THRESHOLDS = {
    'radiotherapy': 0.5,
    'chemotherapy': 0.5,
    'targeted_therapy': 0.2,
    'immunotherapy': 0.2,
    'hormone_therapy': 0.3
}

# ===================== 数据加载 =====================
def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for item in data:
        row = {}
        for k, v in item['X'].items():
            row[f'X_{k}'] = v
        for k, v in item['Y_structured'].items():
            row[f'Y_{k}'] = v
        row['regimen'] = item.get('Y_detail', {}).get('regimen', '')
        row['Y_text'] = item.get('Y_text', '')
        rows.append(row)

    return pd.DataFrame(rows)

# ===================== 软聚类特征生成 =====================
def compute_soft_cluster_features(X_base, kmeans):
    """计算样本到各簇中心的距离，通过 softmax 转为隶属概率"""
    distances = kmeans.transform(X_base)
    soft_probs = softmax(-distances, axis=1)
    return soft_probs

# ===================== 患者检索器（新版） =====================
class PatientRetriever:

    def __init__(self):
        # 加载模型、预处理器、KMeans
        self.models = joblib.load(MODEL_PATH)           # dict: {label: XGBoost}
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.kmeans = joblib.load(KMEANS_PATH)

        self.df = load_data()
        self.X = self.df[[c for c in self.df.columns if c.startswith('X_')]]
        self.Y = self.df[[c for c in self.df.columns if c.startswith('Y_')]]
        # 🔥 显式指定五个治疗标签（排除 Y_text 等非标签列）
        self.label_names = ['radiotherapy', 'chemotherapy', 'targeted_therapy',
                            'immunotherapy', 'hormone_therapy']

        self.feature_columns = self.X.columns.tolist()
        print(f"训练特征列数量: {len(self.feature_columns)}")
        print(f"治疗标签: {self.label_names}")

        print("构建向量空间（基础特征+软聚类特征）...")
        # 生成检索用的特征矩阵（与训练时一致：基础特征 + 软聚类特征）
        X_base = self.preprocessor.transform(self.X)
        X_soft = compute_soft_cluster_features(X_base, self.kmeans)
        self.X_vec = np.column_stack([X_base, X_soft])

        self.knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        self.knn.fit(self.X_vec)

    def _prepare_new_patient_df(self, new_patient_dict):
        """将患者字典转换为完整的 DataFrame，缺失列自动填充默认值"""
        full_dict = {}
        for col in self.feature_columns:
            raw_name = col[2:]  # 去掉 'X_'
            if raw_name in new_patient_dict:
                full_dict[col] = new_patient_dict[raw_name]
            else:
                # 根据字段类型设置合理默认值
                if raw_name in ['age', 'hypertension', 'diabetes', 'obesity', 'adnexal_involvement']:
                    full_dict[col] = 0
                elif raw_name == 'myometrial_invasion_depth':
                    full_dict[col] = None
                elif raw_name == 'menopause':
                    full_dict[col] = 'unknown'
                else:
                    full_dict[col] = 'unknown'
        return pd.DataFrame([full_dict])

    def _transform_patient(self, patient_df):
        """将患者 DataFrame 转换为模型输入特征（基础特征 + 软聚类特征）"""
        X_base = self.preprocessor.transform(patient_df)
        X_soft = compute_soft_cluster_features(X_base, self.kmeans)
        return np.column_stack([X_base, X_soft])

    def retrieve(self, new_patient_dict, top_k=3):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self._transform_patient(new_df)
        dist, idx = self.knn.kneighbors(new_vec, n_neighbors=top_k)
        results = self.df.iloc[idx[0]].copy()
        results['distance'] = dist[0]
        return results

    def predict(self, new_patient_dict):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self._transform_patient(new_df)

        pred_dict = {}
        for label in self.label_names:
            if label == 'surgery':   # 已移除，跳过
                continue
            clf = self.models[label]
            proba = clf.predict_proba(new_vec)[:, 1]
            thr = THRESHOLDS.get(label, 0.5)
            pred = int(proba[0] > thr)
            pred_dict[label] = pred
        return pred_dict

    def build_rag_prompt(self, new_patient, retrieved_df, prediction):
        prompt = []
        prompt.append("你是一名妇科肿瘤专家，请基于以下信息制定治疗方案。\n")

        prompt.append("【当前患者信息】")
        current_row = {f'X_{k}': v for k, v in new_patient.items()}
        current_desc = format_patient_desc(pd.Series(current_row))
        prompt.append(current_desc)

        prompt.append("\n【模型预测的治疗建议】")
        for k, v in prediction.items():
            prompt.append(f"- {k}: {'推荐' if v == 1 else '不推荐'}")

        prompt.append("\n【相似患者案例（按相似度从高到低排序）】")

        for i, (_, row) in enumerate(retrieved_df.head(3).iterrows()):
            order = ["最相似案例", "次相似案例", "第三相似案例"][i]
            prompt.append(f"\n{order}：")

            patient_desc = format_patient_desc(row)
            prompt.append(patient_desc)

            y_text = row.get('Y_text', '')
            if y_text and y_text.strip() and y_text != 'unknown':
                prompt.append(f"  治疗建议原文: {y_text}")



        prompt.append("""
请综合以上信息，给出：
1. **最推荐的真实治疗方案**：必须包含具体化疗方案名称、周期数、放疗技术及剂量参考。优先引用相似案例中已出现的方案细节；
2. **随访计划**：基于患者分期、分子特征和高危因素，给出具体的随访频率、检查项目（如CA125、影像学检查）及持续时间。**请特别关注“治疗建议原文”中可能包含的随访或临床试验安排。**
3. **分子分型指导**：结合患者的p53突变、MMR状态、分子分型等，分析其对预后、化疗敏感性、靶向/免疫治疗机会的影响，并给出相应建议。
4. **个性化调整及注意事项**：针对合并症（高血压、肥胖）及未知因素（如腹主动脉旁淋巴结状态）提出补充检查或治疗调整建议。
请用专业、简洁的语言回答，禁止使用模糊词汇。
            """)
        return "\n".join(prompt)


# ===================== 主程序示例 =====================
if __name__ == "__main__":
    retriever = PatientRetriever()

    new_patient = {
        "age": 58,
        "menopause": "yes",
        "histology_type": "endometrioid",
        "grade": "G2",
        "stage": "II",
        "myometrial_invasion_ratio": ">=50%",
        "myometrial_invasion_depth": None,
        "cervical_involvement": "unknown",
        "lvsi": "positive",
        "lymph_node_pelvic": "1/10",
        "lymph_node_paraaortic": "unknown",
        "peritoneal_cytology": "unknown",
        "adnexal_involvement": 0,
        "p53": "mutant",
        "mmr": "deficient",
        "molecular_subtype": "unknown",
        "hypertension": 1,
        "diabetes": 0,
        "obesity": 1
    }

    print("\n正在检索相似患者...")
    retrieved = retriever.retrieve(new_patient)

    print("模型预测中...")
    pred = retriever.predict(new_patient)

    print("构建 RAG 提示词...")
    prompt = retriever.build_rag_prompt(new_patient, retrieved, pred)

    print("\n" + "="*60)
    print("RAG 提示词 (输入给 vLLM 的内容)")
    print("="*60)
    print(prompt)
    print("="*60)

    print("\n正在调用 本地 vLLM 生成治疗建议...")
    response = call_vllm(prompt)  # 修改为 vLLM 调用

    print("\n" + "="*60)
    print("vLLM 专家建议")
    print("="*60)
    print(response)
    print("="*60)