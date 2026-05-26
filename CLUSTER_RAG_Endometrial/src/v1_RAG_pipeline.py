import json
import numpy as np
import pandas as pd
import re
import joblib
import requests
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ===================== 全局自定义函数（与 train_model.py 保持一致） =====================

def map_stage(arr):
    stage_map = {'IA': 1, 'IB': 2, 'II': 3, 'IIIA': 4, 'IIIB': 5,
                 'IIIC1': 6, 'IIIC2': 7, 'IVA': 8, 'IVB': 9, 'unknown': 0}
    vec_map = np.vectorize(lambda x: stage_map.get(x, 0))
    return vec_map(arr).reshape(-1, 1)

def map_grade(arr):
    grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'unknown': 0}
    vec_map = np.vectorize(lambda x: grade_map.get(x, 0))
    return vec_map(arr).reshape(-1, 1)

def map_ratio(arr):
    ratio_map = {'<50%': 0.25, '>=50%': 0.75, 'unknown': np.nan}
    vec_map = np.vectorize(lambda x: ratio_map.get(x, np.nan))
    return vec_map(arr).reshape(-1, 1)

def simplify_lymph(arr):
    def _f(v):
        if pd.isna(v) or v is None or v == '' or v == 'unknown':
            return 0
        v_str = str(v).strip().lower()
        if v_str == 'negative':
            return 0
        if v_str == 'positive':
            return 1
        if '/' in v_str:
            num = int(v_str.split('/')[0])
            return 1 if num > 0 else 0
        return 0
    vec_f = np.vectorize(_f)
    return vec_f(arr).reshape(-1, 1)

def extract_has_depth(arr):
    def _has(v):
        if pd.isna(v) or v is None or v == '' or v == 'unknown':
            return 0
        if isinstance(v, (int, float)):
            return 1
        if isinstance(v, str) and re.search(r'\d', v):
            return 1
        return 0
    vec_has = np.vectorize(_has)
    return vec_has(arr).reshape(-1, 1)

def binary_flag(arr):
    def _f(v):
        if pd.isna(v) or v is None:
            return 0
        if str(v).strip().lower() == 'positive':
            return 1
        return 0
    vec_f = np.vectorize(_f)
    return vec_f(arr).reshape(-1, 1)


# ===================== DeepSeek API 配置 =====================
DEEPSEEK_API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")  # 从环境变量读取
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_deepseek(prompt, temperature=0.1):
    """调用 DeepSeek API 生成回答"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1000
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API 调用失败: {e}"


# ===================== 路径 =====================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "structured_output.json"
MODEL_PATH = BASE_DIR / "models" / "trained_rf_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


# ===================== 数据加载 =====================
def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for item in data:
        row = {}
        # X 特征
        for k, v in item['X'].items():
            row[f'X_{k}'] = v
        # Y_structured 标签
        for k, v in item['Y_structured'].items():
            row[f'Y_{k}'] = v
        # ✅ 新增：治疗详情
        row['regimen'] = item.get('Y_detail', {}).get('regimen', '')
        row['Y_text'] = item.get('Y_text', '')
        rows.append(row)

    return pd.DataFrame(rows)


# ===================== 患者检索器 =====================
class PatientRetriever:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        self.df = load_data()

        self.X = self.df[[c for c in self.df.columns if c.startswith('X_')]]
        self.Y = self.df[[c for c in self.df.columns if c.startswith('Y_')]]

        # 保存所有训练时的特征列名，供补全使用
        self.feature_columns = self.X.columns.tolist()
        print(f"训练特征列数量: {len(self.feature_columns)}")

        print("构建向量空间...")
        self.X_vec = self.preprocessor.transform(self.X)

        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.knn.fit(self.X_vec)

    def _prepare_new_patient_df(self, new_patient_dict):
        """将患者字典转换为完整的 DataFrame，缺失列自动填充默认值"""
        # 初始化一个全为默认值的字典
        full_dict = {}
        for col in self.feature_columns:
            # 去掉 X_ 前缀得到原始字段名
            raw_name = col[2:]  # 去掉 'X_'
            if raw_name in new_patient_dict:
                full_dict[col] = new_patient_dict[raw_name]
            else:
                # 根据字段类型设置合理的默认值
                if raw_name in ['age', 'hypertension', 'diabetes', 'obesity', 'adnexal_involvement']:
                    full_dict[col] = 0
                elif raw_name == 'myometrial_invasion_depth':
                    full_dict[col] = None
                elif raw_name == 'menopause':
                    full_dict[col] = 'unknown'
                else:
                    full_dict[col] = 'unknown'
        return pd.DataFrame([full_dict])

    def retrieve(self, new_patient_dict, top_k=5):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self.preprocessor.transform(new_df)
        dist, idx = self.knn.kneighbors(new_vec, n_neighbors=top_k)
        results = self.df.iloc[idx[0]].copy()
        results['distance'] = dist[0]
        return results

    def predict(self, new_patient_dict):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self.preprocessor.transform(new_df)
        pred = self.model.predict(new_vec)[0]
        labels = self.Y.columns
        return dict(zip(labels, pred))

    def build_rag_prompt(self, new_patient, retrieved_df, prediction):
        prompt = []
        prompt.append("你是一名妇科肿瘤专家，请基于以下信息制定治疗方案。\n")

        prompt.append("【当前患者信息】")
        for k, v in new_patient.items():
            prompt.append(f"- {k}: {v}")

        prompt.append("\n【模型预测的治疗建议】")
        for k, v in prediction.items():
            prompt.append(f"- {k}: {'推荐' if v == 1 else '不推荐'}")

        prompt.append("\n【相似患者案例（含具体治疗方案）】")
        for i, (_, row) in enumerate(retrieved_df.iterrows()):
            prompt.append(f"\n案例 {i + 1}（相似度距离: {row['distance']:.3f}）")

            # 扩展关键特征列表，包含更多临床决策相关字段
            key_features = [
                'age', 'menopause', 'stage', 'grade', 'histology_type',
                'myometrial_invasion_ratio', 'lvsi', 'lymph_node_pelvic',
                'lymph_node_paraaortic', 'cervical_involvement', 'p53', 'mmr',
                'molecular_subtype', 'hypertension', 'diabetes', 'obesity'
            ]
            for feat in key_features:
                col_name = f'X_{feat}'
                if col_name in row.index:
                    val = row[col_name]
                    if pd.notna(val) and val != '':
                        prompt.append(f"  {feat}: {val}")

            # ✅ 核心改动：输出具体的治疗方案文本
            regimen_text = row.get('regimen', '')
            y_text = row.get('Y_text', '')
            if regimen_text and regimen_text != 'unknown':
                prompt.append(f"  具体方案: {regimen_text}")
            elif y_text and y_text != 'unknown':
                prompt.append(f"  治疗描述: {y_text}")
            else:
                # 如果没有文本方案，则输出标签组合
                treatments = []
                for col in self.Y.columns:
                    if row[col] == 1:
                        treatments.append(col.replace('Y_', ''))
                prompt.append(f"  治疗方案: {', '.join(treatments) if treatments else '无辅助治疗'}")

        prompt.append("""
            请综合以上信息，给出：
            1. 最推荐的治疗组合
            2. 推荐理由（结合相似患者案例与医学指南）
            3. 是否需要个性化调整及注意事项
            请用专业、简洁的语言回答。
            """)
        return "\n".join(prompt)


# ===================== 主程序示例 =====================
if __name__ == "__main__":
    retriever = PatientRetriever()

    # 完整的新患者示例（所有训练时出现的字段都最好写上，缺失的会由程序自动补'unknown'）
    new_patient = {
        "age": 58,
        "menopause": "yes",
        "histology_type": "endometrioid",
        "grade": "G2",
        "stage": "II",
        "myometrial_invasion_ratio": ">=50%",
        "myometrial_invasion_depth": None,  # 可缺失
        "cervical_involvement": "unknown",  # 未知
        "lvsi": "positive",
        "lymph_node_pelvic": "1/10",
        "lymph_node_paraaortic": "unknown",  # 未知
        "peritoneal_cytology": "unknown",  # 未知
        "adnexal_involvement": 0,
        "p53": "mutant",
        "mmr": "deficient",
        "molecular_subtype": "unknown",  # 未知
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
    print("RAG 提示词 (输入给 DeepSeek 的内容)")
    print("="*60)
    print(prompt)
    print("="*60)

    print("\n正在调用 DeepSeek API 生成治疗建议...")
    response = call_deepseek(prompt)

    print("\n" + "="*60)
    print("DeepSeek 专家建议")
    print("="*60)
    print(response)
    print("="*60)