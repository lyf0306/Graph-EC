"""
升级版：train_model_v2_fixed.py
- 保留 KFold 交叉验证
- 保留多标签指标 (Hamming Loss, Jaccard)
- 补齐所有关键临床特征（分子分型、双侧淋巴结、宫颈受累等）
- 对高缺失字段做稳健处理
- 修正：全局自定义函数支持 pickle 序列化
"""

import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    hamming_loss, jaccard_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================== 参数 ========================
RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 100
MAX_DEPTH = 10

SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

INPUT_JSON = DATA_DIR / "structured_output.json"
MODEL_OUTPUT = MODEL_DIR / "trained_rf_model.pkl"
PREPROCESSOR_OUTPUT = MODEL_DIR / "preprocessor.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"输入数据: {INPUT_JSON}")

# ======================== 全局自定义转换函数（支持 pickle） ========================

def map_stage(arr):
    """FIGO 分期序数映射（输入为 1D NumPy 数组）"""
    stage_map = {'IA': 1, 'IB': 2, 'II': 3, 'IIIA': 4, 'IIIB': 5,
                 'IIIC1': 6, 'IIIC2': 7, 'IVA': 8, 'IVB': 9, 'unknown': 0}
    vec_map = np.vectorize(lambda x: stage_map.get(x, 0))
    return vec_map(arr).reshape(-1, 1)

def map_grade(arr):
    """组织学分级序数映射"""
    grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'unknown': 0}
    vec_map = np.vectorize(lambda x: grade_map.get(x, 0))
    return vec_map(arr).reshape(-1, 1)

def map_ratio(arr):
    """肌层浸润比例映射为数值，unknown 返回 np.nan"""
    ratio_map = {'<50%': 0.25, '>=50%': 0.75, 'unknown': np.nan}
    vec_map = np.vectorize(lambda x: ratio_map.get(x, np.nan))
    return vec_map(arr).reshape(-1, 1)

def simplify_lymph(arr):
    """淋巴结状态 → 二值阳性标志 (1=阳性, 0=阴性/未知)"""
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
    """浸润深度 → 是否有数值记录 (1=有, 0=无)"""
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
    """将 lvsi/cytology 等转为 1=positive, 0=其他"""
    def _f(v):
        if pd.isna(v) or v is None:
            return 0
        if str(v).strip().lower() == 'positive':
            return 1
        return 0
    vec_f = np.vectorize(_f)
    return vec_f(arr).reshape(-1, 1)


# ======================== 数据加载 ========================
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for item in data:
        if 'X' not in item:
            continue
        row = {}
        for k, v in item['X'].items():
            row[f'X_{k}'] = v
        for k, v in item['Y_structured'].items():
            row[f'Y_{k}'] = v
        records.append(row)

    return pd.DataFrame(records)


# ======================== 特征工程流水线 ========================
def build_preprocessor():
    """
    构建 ColumnTransformer，使用全局自定义函数。
    """
    # 年龄
    age_col = ['X_age']
    # 有序变量
    stage_col = ['X_stage']
    grade_col = ['X_grade']
    ratio_col = ['X_myometrial_invasion_ratio']
    # 淋巴结（双侧）
    lymph_cols = ['X_lymph_node_pelvic', 'X_lymph_node_paraaortic']
    # 二值标志（LVSI、细胞学、附件）
    binary_flag_cols = ['X_lvsi', 'X_peritoneal_cytology', 'X_adnexal_involvement']
    # 合并症
    comorbidity_cols = ['X_hypertension', 'X_diabetes', 'X_obesity']
    # 深度标志
    depth_col = ['X_myometrial_invasion_depth']
    # 多分类变量（One-Hot）
    categorical_cols = [
        'X_menopause', 'X_histology_type', 'X_cervical_involvement',
        'X_p53', 'X_mmr', 'X_molecular_subtype'
    ]

    # 年龄管道
    age_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 有序变量管道
    stage_pipe = FunctionTransformer(map_stage, validate=False)
    grade_pipe = FunctionTransformer(map_grade, validate=False)
    ratio_pipe = Pipeline([
        ('map', FunctionTransformer(map_ratio, validate=False)),
        ('impute', SimpleImputer(strategy='constant', fill_value=0.25))
    ])

    # 淋巴结管道
    lymph_pipe = FunctionTransformer(simplify_lymph, validate=False)

    # 深度标志管道
    depth_pipe = FunctionTransformer(extract_has_depth, validate=False)

    # 二值标志管道（lvsi, cytology, adnexal）
    binary_pipe = FunctionTransformer(binary_flag, validate=False)

    # 合并症管道
    comorbidity_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # 多分类管道
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # 组装
    preprocessor = ColumnTransformer([
        ('age', age_pipe, age_col),
        ('stage', stage_pipe, stage_col),
        ('grade', grade_pipe, grade_col),
        ('ratio', ratio_pipe, ratio_col),
        ('ln_pelvic', lymph_pipe, ['X_lymph_node_pelvic']),
        ('ln_paraaortic', lymph_pipe, ['X_lymph_node_paraaortic']),
        ('has_depth', depth_pipe, depth_col),
        ('lvsi', binary_pipe, ['X_lvsi']),
        ('cytology', binary_pipe, ['X_peritoneal_cytology']),
        ('adnexal', binary_pipe, ['X_adnexal_involvement']),
        ('comorbidity', comorbidity_pipe, comorbidity_cols),
        ('categorical', cat_pipe, categorical_cols)
    ], remainder='drop')

    return preprocessor


# ======================== 训练与交叉验证 ========================
def train_with_cv(df):
    # 分离特征和标签
    X = df[[c for c in df.columns if c.startswith('X_')]].copy()
    y = df[[c for c in df.columns if c.startswith('Y_')]].copy()
    y.columns = [c.replace('Y_', '') for c in y.columns]

    print(f"\n特征维度(原始): {X.shape[1]}")
    print("标签分布（1=接受）:")
    for col in y.columns:
        print(f"  {col}: {sum(y[col]==1)} ({sum(y[col]==1)/len(y)*100:.1f}%)")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, macro_f1s, hams, jacs = [], [], [], []
    aucs_per_label = {label: [] for label in y.columns}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

        X_train_raw = X.iloc[tr_idx]
        X_test_raw = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test = y.iloc[te_idx]

        pre = build_preprocessor()
        X_train = pre.fit_transform(X_train_raw)
        X_test = pre.transform(X_test_raw)

        if fold == 0:
            try:
                f_names = pre.get_feature_names_out()
                print(f"处理后特征维度: {len(f_names)}")
            except:
                print(f"处理后特征维度: {X_train.shape[1]}")

        model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                class_weight='balanced',
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        ham = hamming_loss(y_test, y_pred)
        jac = jaccard_score(y_test, y_pred, average='samples', zero_division=0)

        accs.append(acc)
        macro_f1s.append(macro_f1)
        hams.append(ham)
        jacs.append(jac)

        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Macro F1     : {macro_f1:.4f}")
        print(f"  Hamming Loss : {ham:.4f}")
        print(f"  Jaccard Score: {jac:.4f}")

        for i, label in enumerate(y.columns):
            proba = y_proba[i]
            if proba.shape[1] == 2:
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba[:, 0]
            try:
                auc = roc_auc_score(y_test[label], proba_pos)
                aucs_per_label[label].append(auc)
            except ValueError:
                aucs_per_label[label].append(np.nan)

    # 汇总
    print("\n" + "="*60)
    print("交叉验证汇总结果 (5-Fold CV)")
    print("="*60)
    print(f"Accuracy      : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1      : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Hamming Loss  : {np.mean(hams):.4f} ± {np.std(hams):.4f}")
    print(f"Jaccard Score : {np.mean(jacs):.4f} ± {np.std(jacs):.4f}")

    print("\n逐标签 AUC (均值±标准差):")
    for label in y.columns:
        vals = [v for v in aucs_per_label[label] if not np.isnan(v)]
        if vals:
            print(f"  {label:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        else:
            print(f"  {label:20s}: N/A")

    # 最终模型（全量数据）
    print("\n训练最终模型（全量数据）...")
    pre_final = build_preprocessor()
    X_all = pre_final.fit_transform(X)

    final_model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    )
    final_model.fit(X_all, y)

    joblib.dump(final_model, MODEL_OUTPUT)
    joblib.dump(pre_final, PREPROCESSOR_OUTPUT)

    print(f"模型已保存至: {MODEL_OUTPUT}")
    print(f"预处理器已保存至: {PREPROCESSOR_OUTPUT}")

    return final_model, pre_final


# ======================== MAIN ========================
if __name__ == "__main__":
    if not INPUT_JSON.exists():
        print(f"错误：数据文件不存在 {INPUT_JSON}")
        exit()

    df = load_data(INPUT_JSON)
    print(f"成功加载 {len(df)} 条记录")

    train_with_cv(df)