"""
v4_train_pipeline.py
=====================
端到端患者相似性检索与多标签治疗方案预测系统

整合内容：
1. 患者相似性检索系统（KMeans + KNN）
2. XGBoost 多标签分类器（5折交叉验证 + 全量重训练）

数据流：
    raw JSON → DataFrame → 特征工程 → 软聚类增强 → [检索索引 + 分类模型]
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================== 全局配置 ========================
RANDOM_STATE = 42          # 全管线复现性种子
N_CLUSTERS = 3             # 软聚类簇数（与检索器一致）
N_SPLITS = 5               # 交叉验证折数
N_ESTIMATORS = 200         # XGBoost 树数量
MAX_DEPTH = 6              # XGBoost 最大深度
LEARNING_RATE = 0.05       # XGBoost 学习率

# 标签阈值（针对类别不平衡调整决策边界）
THRESHOLDS = {
    "radiotherapy": 0.5,
    "chemotherapy": 0.5,
    "targeted_therapy": 0.2,
    "immunotherapy": 0.2,
    "hormone_therapy": 0.3,
}

# 路径解析：支持 "src/" 子目录或项目根目录两种布局
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_JSON = DATA_DIR / "final_patient_vectors_v4.json"

# -------------------- 检索系统输出文件 --------------------
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_retriever_v4.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_retriever_v4.pkl"
FEATURES_PATH = MODEL_DIR / "X_vec_retriever_v4.npy"
KNN_PATH = MODEL_DIR / "knn_index_retriever_v4.pkl"
IDS_PATH = MODEL_DIR / "patient_ids_retriever_v4.pkl"
DF_PATH = MODEL_DIR / "df_retriever_v4.pkl"

# -------------------- 分类器输出文件 --------------------
CLASSIFIER_PATH = MODEL_DIR / "trained_xgb_v4.pkl"

# ======================== 特征工程配置 ========================
# 以下列名必须与 final_patient_vectors_v4.json 的 Schema 严格对应，
# 前缀 X_ 表示在 DataFrame 中自动附加的标识。

CATEGORICAL_COLS = [
    "X_stage_raw",
    "X_figo_version",
    "X_histology_type",
    "X_grade",
    "X_cervical_involvement",
    "X_menopause",
    "X_p53",
    "X_mmr",
    "X_molecular_subtype",
    "X_stage_2023",
    "X_esgo_risk_group",
    "X_myometrial_invasion_ratio",
    "X_lvsi",
    "X_peritoneal_cytology",
]

NUMERICAL_COLS = [
    "X_age",
    "X_myometrial_invasion_depth",
]

# 合并症与二元特征已编码为 0/1/2，无需 OneHot，直接填充缺失即可
COMORBIDITY_COLS = [
    "X_glycemic_status",
    "X_hypertension",
    "X_bmi_status",
    "X_hyperlipidemia",
    "X_anemia",
    "X_hepatic_viral",
    "X_hepatic_dysfunction",
    "X_major_cv_risk",
]

OTHER_BINARY_COLS = [
    "X_lvsi_substantial",
    "X_adnexal_involvement",
]

# 高临床权重特征：这些禁忌症在相似性检索中应被显著放大
WEIGHT_COLS = ["X_major_cv_risk", "X_hepatic_viral"]
WEIGHT_MULTIPLIER = 2.0

# ======================== 数据加载 ========================

def load_data(json_path: Path) -> pd.DataFrame:
    """
    从 final_patient_vectors_v4.json 加载患者记录。

    参数:
        json_path: JSON 文件路径，要求每条记录包含 id, X, Y_structured 字段。

    返回:
        DataFrame，index 为患者 id，列包含特征 X_*、标签 Y_*、
        原始文本 Y_text 以及保留的原始字典 _raw_X（供检索结果展示）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    patient_ids = []

    for item in data:
        pid = item.get("id")
        if pid is None:
            continue  # 跳过无 ID 的异常记录

        patient_ids.append(pid)
        row = {}

        # ---- 主病特征（跳过纯文本描述字段，避免高维稀疏） ----
        for key, val in item["X"].items():
            if key in ("histology_detail", "stage_2023_full", "esgo_recommendation"):
                continue
            row[f"X_{key}"] = val

        # ---- 结构化治疗标签（严格过滤：只保留数值型标签，跳过随访文本等） ----
        for key, val in item.get("Y_structured", {}).items():
            try:
                # 兼容字符串形式的 "0"/"1" 或原生数值
                numeric_val = float(val)
                row[f"Y_{key}"] = numeric_val
            except (ValueError, TypeError):
                # 跳过纯文本字段（如 follow_up、recommendation 等）
                continue

        # ---- 保留原始文本与原始特征字典（检索展示用） ----
        row["Y_text"] = item.get("Y_text", "")
        row["_raw_X"] = item["X"]

        records.append(row)

    df = pd.DataFrame(records, index=patient_ids)
    return df


# ======================== 预处理器构建 ========================

def filter_existing_columns(cols: list[str], df: pd.DataFrame) -> list[str]:
    """过滤掉在 DataFrame 中实际不存在的列，避免 ColumnTransformer 报错。"""
    return [c for c in cols if c in df.columns]


def build_preprocessor(cat_cols: list[str], num_cols: list[str], binary_cols: list[str]) -> ColumnTransformer:
    """
    构建三段式预处理管线。

    设计说明:
        - 类别特征: 缺失填充为 "unknown" 后 OneHot，handle_unknown="ignore" 保证
          生产环境遇到新类别时不崩溃。
        - 数值特征: 中位数填充（对异常值鲁棒）+ Z-score 标准化。
        - 二元/合并症: 仅填充 0，保留原始量纲（已与临床专家确认编码规范）。
    """
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
        ("binary", binary_pipe, binary_cols),
    ], remainder="drop")

    return preprocessor


# ======================== 软聚类特征 ========================

def compute_soft_cluster_features(X_base: np.ndarray, kmeans: KMeans) -> np.ndarray:
    """
    将硬聚类转为软分配（Soft Assignment）。

    原理:
        对每个样本，计算其到各簇中心的距离，经负号后用 softmax 转化为概率分布。
        这样保留了样本处于簇边界的模糊信息，比 one-hot 型硬标签更适合下游分类器。
    """
    distances = kmeans.transform(X_base)          # shape: (n_samples, n_clusters)
    return softmax(-distances, axis=1)            # 距离越近，概率越高


# ======================== 检索系统构建 ========================

def build_retrieval_system(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, KMeans, NearestNeighbors]:
    """
    构建患者相似性检索系统。

    流程:
        1. 提取特征矩阵 X（排除标签与元数据）
        2. 对禁忌症特征手动加权（临床先验）
        3. 三段式预处理 → 基础特征 X_base
        4. KMeans 软聚类 → 软分配特征 X_soft
        5. 拼接为 X_final，拟合 KNN（曼哈顿距离，对混合特征更稳健）

    返回:
        (df_feature, X_final, kmeans_model, knn_model)
    """
    print("\n[1/4] 构建检索系统...")

    # 区分特征 / 标签 / 元数据
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X_raw = df[feature_cols].copy()

    # ---- 临床先验加权 ----
    for col in WEIGHT_COLS:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col] * WEIGHT_MULTIPLIER
            print(f"      已加权 {col} (x{WEIGHT_MULTIPLIER})")

    # ---- 过滤实际存在的列 ----
    cat_cols = filter_existing_columns(CATEGORICAL_COLS, X_raw)
    num_cols = filter_existing_columns(NUMERICAL_COLS, X_raw)
    com_cols = filter_existing_columns(COMORBIDITY_COLS, X_raw)
    bin_cols = filter_existing_columns(OTHER_BINARY_COLS, X_raw)

    print(f"      类别特征数量: {len(cat_cols)}")
    if cat_cols:
        print("        " + ", ".join(cat_cols))
    print(f"      数值特征数量: {len(num_cols)}")
    if num_cols:
        print("        " + ", ".join(num_cols))
    print(f"      合并症特征数量: {len(com_cols)}")
    if com_cols:
        print("        " + ", ".join(com_cols))
    print(f"      其他二元特征数量: {len(bin_cols)}")
    if bin_cols:
        print("        " + ", ".join(bin_cols))

    # 原始输入总特征数（未编码前）
    print(f"      原始输入特征总数: {X_raw.shape[1]}")

    # ---- 拟合预处理器 ----
    preprocessor = build_preprocessor(cat_cols, num_cols, com_cols + bin_cols)
    X_base = preprocessor.fit_transform(X_raw)
    print(f"      预处理后基础特征维度: {X_base.shape[1]} (含独热编码扩展)")

    # ---- KMeans 软聚类 ----
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_base)
    X_soft = compute_soft_cluster_features(X_base, kmeans)
    print(f"      软聚类特征维度: {X_soft.shape[1]}")

    # ---- 最终特征矩阵 ----
    X_final = np.column_stack([X_base, X_soft])
    print(f"      最终特征总维度: {X_final.shape} (基础特征 {X_base.shape[1]} + 软聚类 {X_soft.shape[1]})")

    # ---- KNN 索引 ----
    knn = NearestNeighbors(n_neighbors=10, metric="manhattan", algorithm="auto")
    knn.fit(X_final)
    print("      KNN 索引构建完成 (metric=manhattan)")

    return df, X_final, kmeans, knn, preprocessor


def save_retrieval_artifacts(
    preprocessor: ColumnTransformer,
    kmeans: KMeans,
    X_final: np.ndarray,
    knn: NearestNeighbors,
    df: pd.DataFrame,
) -> None:
    """持久化检索系统所有组件，供 PatientRetrieverV4 加载。"""
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    np.save(FEATURES_PATH, X_final)
    joblib.dump(knn, KNN_PATH)
    joblib.dump(df.index.tolist(), IDS_PATH)
    df.to_pickle(DF_PATH)

    print("\n[检索系统持久化]")
    print(f"      预处理器 : {PREPROCESSOR_PATH.name}")
    print(f"      KMeans   : {KMEANS_PATH.name}")
    print(f"      特征矩阵 : {FEATURES_PATH.name}")
    print(f"      KNN 索引 : {KNN_PATH.name}")
    print(f"      ID 列表  : {IDS_PATH.name}")
    print(f"      DataFrame: {DF_PATH.name}")


# ======================== 分类器训练 ========================

def prepare_label_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    从 DataFrame 中分离特征与标签，并移除阳性率过高的手术标签。

    返回:
        (X, y, label_names)
    """
    X = df[[c for c in df.columns if c.startswith("X_")]].copy()
    y = df[[c for c in df.columns if c.startswith("Y_")]].copy()
    # 强制转数值，无法转换的设为 NaN，随后过滤掉全 NaN 的列（非标签文本）
    y = y.apply(pd.to_numeric, errors='coerce')
    y = y.dropna(axis=1, how='all')
    y.columns = [c.replace("Y_", "") for c in y.columns]

    # surgery 阳性率接近 100%，不提供信息量，移除
    if "surgery" in y.columns:
        y = y.drop(columns=["surgery"])
        print("\n[2/4] 已移除标签: surgery（阳性率过高）")

    # 应用与检索系统一致的加权
    for col in WEIGHT_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce") * WEIGHT_MULTIPLIER

    print("\n标签分布:")
    for col in y.columns:
        pos = int(y[col].sum())
        print(f"      {col:20s}: {pos:3d} 阳性 ({pos / len(y) * 100:.1f}%)")

    return X, y, list(y.columns)


def cross_validate_classifier(
    X: pd.DataFrame, y: pd.DataFrame, label_names: list[str]
) -> dict[str, list[float]]:
    """
    5 折分层交叉验证。

    注意:
        每一折都在训练集独立 fit 预处理器 + KMeans，严防数据泄露。
        这与生产环境（全量 fit）不同，目的是获得无偏的性能估计。
    """
    print("\n[3/4] 开始 5 折交叉验证...")

    # 选择阳性率适中的标签作为分层依据
    stratify_label = "chemotherapy" if "chemotherapy" in y.columns else label_names[0]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 指标收集器
    accs, macro_f1s, hams, jacs = [], [], [], []
    aucs_per_label = {label: [] for label in label_names}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y[stratify_label]), 1):
        print(f"\n  ----- Fold {fold}/{N_SPLITS} -----")

        X_train_raw = X.iloc[tr_idx]
        X_test_raw = X.iloc[te_idx]
        y_train = y.iloc[tr_idx]
        y_test = y.iloc[te_idx]

        # ---- 折内重新 fit 预处理器（防止泄露） ----
        cat_cols = filter_existing_columns(CATEGORICAL_COLS, X_train_raw)
        num_cols = filter_existing_columns(NUMERICAL_COLS, X_train_raw)
        com_cols = filter_existing_columns(COMORBIDITY_COLS, X_train_raw)
        bin_cols = filter_existing_columns(OTHER_BINARY_COLS, X_train_raw)

        pre = build_preprocessor(cat_cols, num_cols, com_cols + bin_cols)
        X_train_base = pre.fit_transform(X_train_raw)
        X_test_base = pre.transform(X_test_raw)

        # ---- 折内重新 fit KMeans（防止泄露） ----
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train_base)
        train_soft = compute_soft_cluster_features(X_train_base, kmeans)
        test_soft = compute_soft_cluster_features(X_test_base, kmeans)

        X_train = np.column_stack([X_train_base, train_soft])
        X_test = np.column_stack([X_test_base, test_soft])

        if fold == 1:
            print(f"      基础特征维度: {X_train_base.shape[1]}")
            print(f"      总特征维度  : {X_train.shape[1]}")

        # ---- 逐标签训练 XGBoost（处理类别不平衡） ----
        y_pred_list, y_proba_list = [], []

        for label in label_names:
            pos = y_train[label].sum()
            neg = len(y_train) - pos
            # 自动计算正负样本权重，缓解类别不平衡
            scale_pos_weight = neg / (pos + 1e-5)

            clf = XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            clf.fit(X_train, y_train[label])

            proba = clf.predict_proba(X_test)[:, 1]
            thr = THRESHOLDS.get(label, 0.5)
            pred = (proba > thr).astype(int)

            y_pred_list.append(pred)
            y_proba_list.append(proba)

        y_pred = np.column_stack(y_pred_list)
        y_proba = np.column_stack(y_proba_list)

        # ---- 指标计算 ----
        accs.append(accuracy_score(y_test, y_pred))
        macro_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        hams.append(hamming_loss(y_test, y_pred))
        jacs.append(jaccard_score(y_test, y_pred, average="samples", zero_division=0))

        print(f"      Accuracy : {accs[-1]:.4f} | Macro F1: {macro_f1s[-1]:.4f} | "
              f"Hamming: {hams[-1]:.4f} | Jaccard: {jacs[-1]:.4f}")

        # 逐标签 AUC 与详细报告
        for i, label in enumerate(label_names):
            print(f"\n      Label: {label} (Fold {fold})")
            report = classification_report(
                y_test[label], y_pred[:, i], target_names=["No", "Yes"], zero_division=0
            )
            for line in report.splitlines():
                print(f"        {line}")
            try:
                auc = roc_auc_score(y_test[label], y_proba[:, i])
                aucs_per_label[label].append(auc)
            except ValueError:
                aucs_per_label[label].append(np.nan)

    # ---- 汇总输出 ----
    print("\n" + "=" * 60)
    print("交叉验证汇总 (XGBoost + 软聚类)")
    print("=" * 60)
    print(f"Accuracy      : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1      : {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Hamming Loss  : {np.mean(hams):.4f} ± {np.std(hams):.4f}")
    print(f"Jaccard Score : {np.mean(jacs):.4f} ± {np.std(jacs):.4f}")
    print("\n逐标签 AUC:")
    for label in label_names:
        vals = [v for v in aucs_per_label[label] if not np.isnan(v)]
        if vals:
            print(f"  {label:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        else:
            print(f"  {label:20s}: N/A")

    return aucs_per_label


def train_final_classifier(
    X: pd.DataFrame, y: pd.DataFrame, label_names: list[str], preprocessor: ColumnTransformer
) -> dict[str, XGBClassifier]:
    """
    在全量数据上训练最终分类器，复用已拟合的预处理器与 KMeans（与检索系统一致）。

    参数:
        X: 全量特征（已加权）
        y: 全量标签
        label_names: 标签名列表
        preprocessor: 已在全量数据上 fit 的预处理器

    返回:
        标签 -> 模型 的字典
    """
    print("\n[4/4] 在全量数据上训练最终分类器...")

    # 使用检索系统已 fit 的预处理器，保证生产一致性
    X_base = preprocessor.transform(X)

    # 使用检索系统已 fit 的 KMeans，保证聚类空间一致
    kmeans = joblib.load(KMEANS_PATH)
    X_soft = compute_soft_cluster_features(X_base, kmeans)
    X_all = np.column_stack([X_base, X_soft])

    final_models = {}
    for label in label_names:
        pos = y[label].sum()
        neg = len(y) - pos
        scale_pos_weight = neg / (pos + 1e-5)

        clf = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_all, y[label])
        final_models[label] = clf

    joblib.dump(final_models, CLASSIFIER_PATH)
    print(f"\n模型已保存至: {CLASSIFIER_PATH}")
    return final_models


# ======================== 主流程 ========================

def main() -> None:
    """端到端训练管线入口。"""
    print(f"输入数据: {INPUT_JSON}")
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"数据文件未找到: {INPUT_JSON}，请确认路径。")

    # ---- 1. 加载原始数据 ----
    df = load_data(INPUT_JSON)
    print(f"成功加载 {len(df)} 条记录，原始列数: {len(df.columns)}")

    # 打印所有列名，帮助确认特征与标签的正确加载
    print("\n所有数据列名预览:")
    for col in df.columns:
        print(f"   {col}")


    # ---- 2. 构建并保存检索系统 ----
    df, X_final, kmeans, knn, preprocessor = build_retrieval_system(df)
    save_retrieval_artifacts(preprocessor, kmeans, X_final, knn, df)

    # ---- 3. 准备分类数据 ----
    X_cls, y_cls, label_names = prepare_label_data(df)

    # ---- 4. 交叉验证（获得无偏性能估计） ----
    cross_validate_classifier(X_cls, y_cls, label_names)

    # ---- 5. 全量训练最终模型（复用检索系统的 preprocessor） ----
    # 注意：X_cls 已经过加权，preprocessor 已在 build_retrieval_system 中 fit
    train_final_classifier(X_cls, y_cls, label_names, preprocessor)

    print("\n" + "=" * 60)
    print("✅ 全量训练管线完成！产出文件：")
    print("=" * 60)
    for p in [
        PREPROCESSOR_PATH, KMEANS_PATH, FEATURES_PATH,
        KNN_PATH, IDS_PATH, DF_PATH, CLASSIFIER_PATH,
    ]:
        print(f"   {p}")


if __name__ == "__main__":
    main()