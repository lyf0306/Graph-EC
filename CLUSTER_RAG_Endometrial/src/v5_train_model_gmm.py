# -*- coding: utf-8 -*-
"""
v5_train_model_gmm.py
=====================
GMM 版端到端患者相似性检索与多标签治疗方案预测系统（论文 5.3 对齐）

与 v4 的关键差异：
1. 聚类算法：KMeans(3) + softmax(-距离)  -->  GaussianMixture(covariance_type='full') + predict_proba 后验概率
2. BIC 选参：K ∈ [K_MIN, K_MAX] 网格搜索，论文预期 K=4（BIC 最低）
3. 新增统计验证模块（论文 5.3）：四簇画像表、ANOVA Top-10、卡方检验 + Cramér's V、BIC/PCA/t-SNE 图

【生产接入点说明（后续任务，本脚本不修改生产文件）】
- 生产检索器 PatientRetrieverV4._transform_patient 使用 kmeans.transform + softmax；
  GaussianMixture 没有 .transform()，切换时需改为 gmm.predict_proba(X_base)。
- api/services/resource_manager.py 加载 kmeans_retriever_v4.pkl 处需改为 gmm_retriever_v5.pkl。

数据流：
    raw JSON -> DataFrame -> 特征工程 -> [BIC 选 K] -> GMM 软聚类增强 -> [检索索引 + 分类模型]
    + 统计验证：四簇画像 / ANOVA / 卡方 / PCA-tSNE

用法：
    python v5_train_model_gmm.py --data <路径> [--mode train|validate|all] [--robust]
    python v5_train_model_gmm.py --data smoke --mode all   # 无真实数据时的冒烟测试
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "缺少 xgboost，请先安装：pip install xgboost  "
        "（本脚本还需 scikit-learn / scipy / pandas / numpy / joblib / matplotlib）"
    )

try:
    import joblib
except ImportError:
    raise ImportError("缺少 joblib，请先安装：pip install joblib")

# ---- matplotlib（Agg 后端，无需显示环境） ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体兜底
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ======================== 全局配置 ========================
RANDOM_STATE = 42          # 全管线复现性种子（论文：所有实验固定随机种子 42）
K_MIN = 2                  # BIC 候选聚类数下界（论文：K=2~10）
K_MAX = 10                 # BIC 候选聚类数上界
GMM_COVARIANCE_TYPE = "full"   # 论文：协方差类型 full covariance，刻画非球形分布簇
GMM_N_INIT = 10
GMM_REG_COVAR = 1e-6
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
RESULTS_DIR = PROJECT_ROOT / "results_v5"

DEFAULT_INPUT_JSON = DATA_DIR / "final_patient_vectors_v4.json"

# -------------------- 检索系统输出文件（v5，不覆盖 v4） --------------------
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_retriever_v5.pkl"
GMM_PATH = MODEL_DIR / "gmm_retriever_v5.pkl"
FEATURES_PATH = MODEL_DIR / "X_vec_retriever_v5.npy"
KNN_PATH = MODEL_DIR / "knn_index_retriever_v5.pkl"
IDS_PATH = MODEL_DIR / "patient_ids_retriever_v5.pkl"
DF_PATH = MODEL_DIR / "df_retriever_v5.pkl"
CLUSTER_META_PATH = MODEL_DIR / "cluster_meta_v5.json"

# -------------------- 分类器输出文件 --------------------
CLASSIFIER_PATH = MODEL_DIR / "trained_xgb_v5.pkl"

# ======================== 特征工程配置 ========================
# 与 v4 完全一致（26 个原始特征），保证与论文"26 个原始临床特征"对应
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

# 高临床权重特征：这些禁忌症在相似性检索中应被显著放大（论文：2 倍权重）
WEIGHT_COLS = ["X_major_cv_risk", "X_hepatic_viral"]
WEIGHT_MULTIPLIER = 2.0

TEXT_SKIP_COLS = ("histology_detail", "stage_2023_full", "esgo_recommendation")


# ======================== 数据加载 ========================

def load_data(json_path: Path) -> pd.DataFrame:
    """
    从 JSON 加载患者记录（增强容错版）。

    兼容顶层为 list 或 dict({id: record}) 两种结构；单条记录解析失败仅告警跳过，
    不让全流程崩溃。返回 DataFrame：index 为患者 id，列包含特征 X_*、标签 Y_*、
    原始文本 Y_text 以及保留的原始字典 _raw_X。
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"数据文件未找到: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 兼容 list 或 dict 两种顶层结构
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"JSON 顶层结构不支持: {type(raw)}，期望 list 或 dict")

    records = []
    patient_ids = []
    skipped = 0

    for item in items:
        try:
            if not isinstance(item, dict):
                raise ValueError("记录非 dict")
            pid = item.get("id")
            x_dict = item.get("X")
            if not isinstance(x_dict, dict):
                skipped += 1
                continue
            if pid is None:
                skipped += 1
                continue

            patient_ids.append(pid)
            row = {}

            # ---- 主病特征（跳过纯文本描述字段，避免高维稀疏） ----
            for key, val in x_dict.items():
                if key in TEXT_SKIP_COLS:
                    continue
                row[f"X_{key}"] = val

            # ---- 结构化治疗标签（只保留数值型标签） ----
            for key, val in (item.get("Y_structured") or {}).items():
                try:
                    row[f"Y_{key}"] = float(val)
                except (ValueError, TypeError):
                    continue

            # ---- 保留原始文本与原始特征字典 ----
            row["Y_text"] = item.get("Y_text", "")
            row["_raw_X"] = x_dict
            records.append(row)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"  [告警] 跳过异常记录: {e}")

    df = pd.DataFrame(records, index=patient_ids)
    print(f"成功加载 {len(df)} 条记录（跳过 {skipped} 条）")
    return df


def generate_synthetic_data(n: int = 300) -> pd.DataFrame:
    """
    生成符合 v4 schema 的合成患者数据（冒烟测试用，无真实数据时验证全流程）。
    """
    rng = np.random.default_rng(RANDOM_STATE)

    cat_choices = {
        "stage_raw": ["IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB", "unknown"],
        "figo_version": ["2009", "2023", "unknown"],
        "histology_type": ["endometrioid", "serous", "clear_cell", "carcinosarcoma", "mixed", "unknown"],
        "grade": ["G1", "G2", "G3", "unknown"],
        "cervical_involvement": ["none", "glandular", "stromal", "unknown"],
        "menopause": ["yes", "no", "unknown"],
        "p53": ["wild", "mutant", "unknown"],
        "mmr": ["proficient", "deficient", "unknown"],
        "molecular_subtype": ["POLEmut", "MMRd", "NSMP", "p53abn", "unknown"],
        "stage_2023": ["IA", "IB", "II", "III", "IV", "unknown"],
        "esgo_risk_group": ["Low", "Intermediate", "High-Intermediate", "High", "Uncertain"],
        "myometrial_invasion_ratio": ["<50%", ">=50%", "unknown"],
        "lvsi": ["positive", "negative", "unknown"],
        "peritoneal_cytology": ["negative", "positive", "unknown"],
    }
    labels = ["radiotherapy", "chemotherapy", "targeted_therapy", "immunotherapy", "hormone_therapy"]

    records = []
    ids = []
    for i in range(n):
        row = {}
        for key, choices in cat_choices.items():
            row[f"X_{key}"] = choices[rng.integers(0, len(choices))]
        row["X_age"] = float(rng.integers(30, 80))
        row["X_myometrial_invasion_depth"] = float(rng.integers(0, 30)) if rng.random() > 0.1 else np.nan
        for key in COMORBIDITY_COLS:
            row[f"X_{key.replace('X_', '')}"] = float(rng.integers(0, 3))
        row["X_lvsi_substantial"] = float(rng.integers(0, 2))
        row["X_adnexal_involvement"] = float(rng.integers(0, 2))
        for lab in labels:
            row[f"Y_{lab}"] = float(rng.integers(0, 2))
        row["Y_text"] = "合成治疗建议文本"
        row["_raw_X"] = {k[2:]: v for k, v in row.items() if k.startswith("X_")}
        records.append(row)
        ids.append(f"SYNTH_{i:04d}")

    return pd.DataFrame(records, index=ids)


# ======================== 预处理器构建 ========================

def filter_existing_columns(cols: list[str], df: pd.DataFrame) -> list[str]:
    """过滤掉在 DataFrame 中实际不存在的列，避免 ColumnTransformer 报错。"""
    return [c for c in cols if c in df.columns]


def build_preprocessor(cat_cols: list[str], num_cols: list[str], binary_cols: list[str]) -> ColumnTransformer:
    """
    构建三段式预处理管线（与 v4 完全一致）。
        - 类别特征: 缺失填充 "unknown" 后 OneHot，handle_unknown="ignore"
        - 数值特征: 中位数填充 + Z-score 标准化
        - 二元/合并症: 仅填充 0，保留原始量纲（有序变量不做标准化，论文 5.2.6）
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


# ======================== GMM 软聚类 ========================

def select_k_by_bic(
    X_base: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    covariance_type: str = GMM_COVARIANCE_TYPE,
    random_state: int = RANDOM_STATE,
) -> tuple[int, list[int], list[float], list[float], GaussianMixture]:
    """
    对 K∈[k_min, k_max] 各拟合一次 GMM，用 BIC 选最优成分数（论文 5.3.1）。

    返回:
        (best_k, ks, bics, aics, best_gmm)
    奇异协方差时提升 reg_covar 重试；某个 K 失败记 NaN 跳过；全部失败回退 K=3。
    """
    if np.isnan(X_base).any():
        raise ValueError("X_base 含 NaN，GMM 无法拟合。请检查预处理（应有缺失填充）。")

    ks = list(range(k_min, k_max + 1))
    bics, aics = [], []
    best_bic = float("inf")
    best_gmm = None
    best_k = k_min

    for k in ks:
        gmm = None
        for reg in (GMM_REG_COVAR, 1e-4, 1e-2, 1e-1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    random_state=random_state,
                    n_init=GMM_N_INIT,
                    reg_covar=reg,
                )
                gmm.fit(X_base)
                break
            except Exception:
                gmm = None
        if gmm is None:
            print(f"  K={k:2d}: BIC= NaN（拟合失败，跳过）")
            bics.append(np.nan)
            aics.append(np.nan)
            continue

        bic = gmm.bic(X_base)
        aic = gmm.aic(X_base)
        bics.append(bic)
        aics.append(aic)
        print(f"  K={k:2d}: BIC={bic:>15.2f}  AIC={aic:>15.2f}")
        if not np.isnan(bic) and bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    if best_gmm is None:
        print("[警告] 所有 K 的 GMM 均拟合失败，回退到 K=3。")
        best_k = 3
        best_gmm = GaussianMixture(
            n_components=best_k, covariance_type=covariance_type,
            random_state=random_state, n_init=GMM_N_INIT, reg_covar=GMM_REG_COVAR,
        )
        best_gmm.fit(X_base)
        best_bic = best_gmm.bic(X_base)

    # BIC 单调无拐点提示
    finite_bics = [b for b in bics if not np.isnan(b)]
    if len(finite_bics) >= 2 and finite_bics == sorted(finite_bics):
        print("[提示] BIC 在整个区间单调下降（无拐点），已取最小值处 K；可结合肘部法则/临床可解释性人工复核。")

    print(f"  >> 最优 K = {best_k}（BIC = {best_bic:.2f}）")
    return best_k, ks, bics, aics, best_gmm


def compute_gmm_soft_features(X_base: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """
    软分配特征 = GMM 后验概率（论文 2.4.2：收敛后每个样本获得一个概率向量，
    作为软特征增强下游模型）。替代 v4 的 softmax(-kmeans.transform(X_base))。
    """
    return gmm.predict_proba(X_base)


# ======================== 检索系统构建 ========================

def build_retrieval_system_gmm(
    df: pd.DataFrame,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, GaussianMixture, NearestNeighbors, ColumnTransformer, int, dict]:
    """
    构建 GMM 版患者相似性检索系统。

    流程:
        1. 提取特征矩阵 X（排除标签与元数据）
        2. 对禁忌症特征手动加权（临床先验，2 倍权重）
        3. 三段式预处理 → 基础特征 X_base
        4. BIC 选 K（或固定 n_components）→ GMM 软聚类 → 软分配特征 X_soft
        5. 拼接为 X_final，拟合 KNN（曼哈顿距离，对混合特征更稳健）

    返回:
        (df, X_final, gmm, knn, preprocessor, best_k, cluster_meta)
    """
    print("\n[1/5] 构建 GMM 检索系统...")

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

    print(f"      原始输入特征总数: {X_raw.shape[1]}")
    print(f"      类别={len(cat_cols)} 数值={len(num_cols)} 合并症={len(com_cols)} 其他二元={len(bin_cols)}")

    # ---- 拟合预处理器 ----
    preprocessor = build_preprocessor(cat_cols, num_cols, com_cols + bin_cols)
    X_base = preprocessor.fit_transform(X_raw)
    print(f"      预处理后基础特征维度: {X_base.shape[1]} (含独热编码扩展)")

    # ---- BIC 选 K + GMM 软聚类 ----
    if n_components is None:
        best_k, ks, bics, aics, gmm = select_k_by_bic(X_base, k_min=k_min, k_max=k_max)
        cluster_meta = {"best_k": best_k, "ks": ks, "bics": bics, "aics": aics,
                        "covariance_type": GMM_COVARIANCE_TYPE, "random_state": RANDOM_STATE}
    else:
        best_k = int(n_components)
        gmm = GaussianMixture(
            n_components=best_k, covariance_type=GMM_COVARIANCE_TYPE,
            random_state=RANDOM_STATE, n_init=GMM_N_INIT, reg_covar=GMM_REG_COVAR,
        )
        gmm.fit(X_base)
        cluster_meta = {"best_k": best_k, "ks": [best_k], "bics": [gmm.bic(X_base)],
                        "aics": [gmm.aic(X_base)],
                        "covariance_type": GMM_COVARIANCE_TYPE, "random_state": RANDOM_STATE}

    X_soft = compute_gmm_soft_features(X_base, gmm)
    print(f"      GMM 软聚类特征维度: {X_soft.shape[1]}")

    # ---- 最终特征矩阵 ----
    X_final = np.column_stack([X_base, X_soft])
    print(f"      最终特征总维度: {X_final.shape} (基础 {X_base.shape[1]} + 软聚类 {X_soft.shape[1]})")

    # ---- KNN 索引（曼哈顿距离，论文 2.4.3） ----
    knn = NearestNeighbors(n_neighbors=10, metric="manhattan", algorithm="auto")
    knn.fit(X_final)
    print("      KNN 索引构建完成 (metric=manhattan)")

    return df, X_final, gmm, knn, preprocessor, best_k, cluster_meta


def save_retrieval_artifacts_v5(
    preprocessor: ColumnTransformer,
    gmm: GaussianMixture,
    X_final: np.ndarray,
    knn: NearestNeighbors,
    df: pd.DataFrame,
    cluster_meta: dict,
) -> None:
    """持久化检索系统所有组件（v5 命名，不覆盖 v4）。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(gmm, GMM_PATH)
    np.save(FEATURES_PATH, X_final)
    joblib.dump(knn, KNN_PATH)
    joblib.dump(df.index.tolist(), IDS_PATH)
    df.to_pickle(DF_PATH)
    with open(CLUSTER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_meta, f, ensure_ascii=False, indent=2)

    print("\n[检索系统持久化 v5]")
    print(f"      预处理器 : {PREPROCESSOR_PATH.name}")
    print(f"      GMM      : {GMM_PATH.name}")
    print(f"      特征矩阵 : {FEATURES_PATH.name}")
    print(f"      KNN 索引 : {KNN_PATH.name}")
    print(f"      ID 列表  : {IDS_PATH.name}")
    print(f"      DataFrame: {DF_PATH.name}")
    print(f"      聚类元数据: {CLUSTER_META_PATH.name}")


# ======================== 分类器训练 ========================

def prepare_label_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    从 DataFrame 中分离特征与标签，并移除阳性率过高的手术标签（与 v4 一致）。
    """
    X = df[[c for c in df.columns if c.startswith("X_")]].copy()
    y = df[[c for c in df.columns if c.startswith("Y_")]].copy()
    y = y.apply(pd.to_numeric, errors='coerce')
    y = y.dropna(axis=1, how='all')
    y.columns = [c.replace("Y_", "") for c in y.columns]

    if "surgery" in y.columns:
        y = y.drop(columns=["surgery"])
        print("\n[2/5] 已移除标签: surgery（阳性率过高）")

    # 应用与检索系统一致的加权
    for col in WEIGHT_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce") * WEIGHT_MULTIPLIER

    print("\n标签分布:")
    for col in y.columns:
        pos = int(y[col].sum())
        print(f"      {col:20s}: {pos:3d} 阳性 ({pos / len(y) * 100:.1f}%)")

    return X, y, list(y.columns)


def cross_validate_classifier_gmm(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_names: list[str],
    n_components: int | None = None,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> dict[str, list[float]]:
    """
    5 折分层交叉验证。

    注意:
        每一折都在训练集独立 fit 预处理器 + GMM（含折内 BIC 选 K），严防数据泄露。
        与生产环境（全量 fit）不同，目的是获得无偏的性能估计。
    """
    print("\n[3/5] 开始 5 折交叉验证...")

    stratify_label = "chemotherapy" if "chemotherapy" in y.columns else label_names[0]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, macro_f1s, hams, jacs = [], [], [], []
    aucs_per_label = {label: [] for label in label_names}
    fold_ks = []

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

        # ---- 折内重新 fit GMM（防止泄露）；默认折内 BIC 选 K ----
        if n_components is None:
            k_fold, _, _, _, gmm = select_k_by_bic(X_train_base, k_min=k_min, k_max=k_max)
        else:
            k_fold = int(n_components)
            gmm = GaussianMixture(
                n_components=k_fold, covariance_type=GMM_COVARIANCE_TYPE,
                random_state=RANDOM_STATE, n_init=GMM_N_INIT, reg_covar=GMM_REG_COVAR,
            )
            gmm.fit(X_train_base)
        fold_ks.append(k_fold)

        train_soft = compute_gmm_soft_features(X_train_base, gmm)
        test_soft = compute_gmm_soft_features(X_test_base, gmm)

        X_train = np.column_stack([X_train_base, train_soft])
        X_test = np.column_stack([X_test_base, test_soft])

        if fold == 1:
            print(f"      基础特征维度: {X_train_base.shape[1]}  总特征维度: {X_train.shape[1]}")

        # ---- 逐标签训练 XGBoost（处理类别不平衡） ----
        y_pred_list, y_proba_list = [], []

        for label in label_names:
            pos = y_train[label].sum()
            neg = len(y_train) - pos
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

        for i, label in enumerate(label_names):
            try:
                auc = roc_auc_score(y_test[label], y_proba[:, i])
                aucs_per_label[label].append(auc)
            except ValueError:
                aucs_per_label[label].append(np.nan)

    # ---- 汇总输出 ----
    print("\n" + "=" * 60)
    print("交叉验证汇总 (XGBoost + GMM 软聚类)")
    print("=" * 60)
    print(f"折内最优 K: {fold_ks}")
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


def train_final_classifier_gmm(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_names: list[str],
    preprocessor: ColumnTransformer,
    gmm: GaussianMixture,
) -> dict[str, XGBClassifier]:
    """
    在全量数据上训练最终分类器，复用已拟合的预处理器与 GMM（与检索系统一致）。
    """
    print("\n[4/5] 在全量数据上训练最终分类器...")

    X_base = preprocessor.transform(X)
    X_soft = compute_gmm_soft_features(X_base, gmm)
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
    print(f"模型已保存至: {CLASSIFIER_PATH}")
    return final_models


# ======================== 统计验证（论文 5.3） ========================

def _expanded_to_original_col(name: str) -> str:
    """
    把预处理展开后的列名映射回原始 26 特征名。
        'cat__X_stage_raw_IA' -> 'X_stage_raw'
        'num__X_age'          -> 'X_age'
        'binary__X_glycemic_status' -> 'X_glycemic_status'
    """
    parts = str(name).split("__", 1)
    if len(parts) == 1:
        return str(name)
    transformer, body = parts
    if transformer == "cat":
        # body = 'X_<col>_<value>'，去掉最后一个 _value
        idx = body.rfind("_")
        if idx > 0:
            return body[:idx]
    return body


def get_soft_labels(preprocessor, df: pd.DataFrame, gmm) -> np.ndarray:
    """对全量数据生成聚类硬标签（GMM predict，即后验概率 argmax）。"""
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X_raw = df[feature_cols].copy()
    for col in WEIGHT_COLS:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col] * WEIGHT_MULTIPLIER
    X_base = preprocessor.transform(X_raw)
    return gmm.predict(X_base)


def compute_comorbidity_score(df: pd.DataFrame) -> pd.Series:
    """合并症评分 = 8 项合并症字段求和（论文表 5.6"合并症评分"）。"""
    cols = [c for c in COMORBIDITY_COLS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def compute_other_risk_score(df: pd.DataFrame) -> pd.Series:
    """其他风险评分 = lvsi_substantial + adnexal_involvement（论文表 5.6/5.7）。"""
    cols = [c for c in OTHER_BINARY_COLS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def build_cluster_profile_table(df: pd.DataFrame, labels: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """
    表 5.6 四簇临床画像：各簇例数%、年龄、肌层浸润深度、合并症评分、其他风险评分。
    """
    print("\n[验证] 四簇临床画像（表 5.6）")
    labels_series = pd.Series(labels, index=df.index, name="cluster")
    df_aug = df.copy()
    df_aug["cluster"] = labels_series
    df_aug["comorbidity_score"] = compute_comorbidity_score(df)
    df_aug["other_risk_score"] = compute_other_risk_score(df)

    n_total = len(df)
    rows = []
    for c in sorted(set(labels)):
        sub = df_aug[df_aug["cluster"] == c]
        row = {
            "簇": c,
            "例数": int(len(sub)),
            "例数(%)": f"{len(sub) / n_total * 100:.1f}",
            "年龄均值": sub["X_age"].pipe(pd.to_numeric, errors="coerce").mean(),
            "肌层浸润深度均值(mm)": sub["X_myometrial_invasion_depth"].pipe(pd.to_numeric, errors="coerce").mean(),
            "合并症评分均值": sub["comorbidity_score"].mean(),
            "其他风险评分均值": sub["other_risk_score"].mean(),
        }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("簇")
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table_cluster_profile_v5.csv", encoding="utf-8-sig")
    print(f"  已保存 -> {out_dir / 'table_cluster_profile_v5.csv'}")
    return table


def anova_top_features(
    preprocessor, df: pd.DataFrame, labels: np.ndarray, n_top: int = 10, out_dir: Path = None
) -> pd.DataFrame:
    """
    表 5.7 簇分离驱动特征 Top-N（单因素方差分析 ANOVA F-test）。
    按展开列计算 F 值，再聚合回原始 26 特征（论文按临床特征叙事）。
    """
    print(f"\n[验证] 簇分离驱动特征 Top-{n_top}（ANOVA F-test，表 5.7）")
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X_raw = df[feature_cols].copy()
    for col in WEIGHT_COLS:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col] * WEIGHT_MULTIPLIER
    X_base = preprocessor.transform(X_raw)
    names = preprocessor.get_feature_names_out()

    f_vals, p_vals = f_classif(X_base, labels)
    f_vals = np.where(np.isnan(f_vals), 0.0, f_vals)   # 常数列 F 为 NaN -> 0

    # 聚合回原始特征
    agg = {}
    for name, f, p in zip(names, f_vals, p_vals):
        orig = _expanded_to_original_col(name)
        cur_f, cur_p = agg.get(orig, (0.0, 1.0))
        agg[orig] = (cur_f + float(f), min(cur_p, float(p)))

    rows = [
        {"特征": orig, "F值": f, "p值": p}
        for orig, (f, p) in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:n_top]
    ]
    table = pd.DataFrame(rows).reset_index(drop=True)
    table.index += 1  # 排名从 1 开始
    table.index.name = "排名"
    print(table.to_string())
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "table_anova_top10_v5.csv", encoding="utf-8-sig")
        print(f"  已保存 -> {out_dir / 'table_anova_top10_v5.csv'}")
    return table


def treatment_acceptance_analysis(df: pd.DataFrame, labels: np.ndarray, label_names: list[str], out_dir: Path) -> pd.DataFrame:
    """
    表 5.8 各簇治疗方案接受率 + 卡方检验 + Cramér's V。
    """
    print("\n[验证] 分型与治疗决策关联（表 5.8）")
    label_names = [l for l in label_names if f"Y_{l}" in df.columns]
    labels_series = pd.Series(labels, index=df.index, name="cluster")

    rows = []
    n_total = len(df)
    for lab in label_names:
        col = f"Y_{lab}"
        y = pd.to_numeric(df[col], errors="coerce")
        rates = {}
        for c in sorted(set(labels)):
            mask = labels_series == c
            sub = y[mask]
            rate = float(sub.mean() * 100) if len(sub) > 0 and not np.isnan(sub.mean()) else 0.0
            rates[f"簇{c}接受率(%)"] = rate

        # 2 x K 列联表：行 = 未接受/接受，列 = 簇
        table_2k = np.zeros((2, len(sorted(set(labels)))))
        for c in sorted(set(labels)):
            mask = labels_series == c
            sub = y[mask].dropna()
            if len(sub) == 0:
                continue
            table_2k[1, c] = sub.sum()
            table_2k[0, c] = len(sub) - sub.sum()

        try:
            chi2, p, _, _ = chi2_contingency(table_2k)
            cramers_v = float(np.sqrt(chi2 / max(n_total * 1.0, 1e-9)))
        except ValueError:
            chi2, p, cramers_v = np.nan, np.nan, np.nan

        rows.append({
            "治疗": lab, **rates,
            "chi2": round(chi2, 1) if not np.isnan(chi2) else np.nan,
            "p值": p if not np.isnan(p) else np.nan,
            "Cramér's V": round(cramers_v, 3) if not np.isnan(cramers_v) else np.nan,
        })

    table = pd.DataFrame(rows)
    print(table.to_string())
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table_treatment_acceptance_v5.csv", encoding="utf-8-sig")
    print(f"  已保存 -> {out_dir / 'table_treatment_acceptance_v5.csv'}")
    return table


def plot_bic_curve(ks, bics, out_dir: Path):
    """图 5.1 BIC 变化曲线。"""
    print("\n[验证] 绘制 BIC 变化曲线...")
    plt.figure(figsize=(8, 5))
    plt.plot(ks, bics, "o-", color="#1f77b4")
    plt.xlabel("聚类数 K")
    plt.ylabel("BIC")
    plt.title("GMM 聚类 BIC 变化曲线")
    plt.grid(alpha=0.3)
    best_k = ks[int(np.nanargmin(bics))]
    plt.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bic_curve_v5.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  已保存 -> {path}")


def plot_projection(preprocessor, df: pd.DataFrame, labels: np.ndarray, out_dir: Path):
    """
    图 5.2 GMM 聚类结果二维投影可视化（PCA + t-SNE）。
    """
    print("\n[验证] 绘制 PCA / t-SNE 二维投影...")
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X_raw = df[feature_cols].copy()
    for col in WEIGHT_COLS:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col] * WEIGHT_MULTIPLIER
    X_base = preprocessor.transform(X_raw)

    n_clusters = len(set(labels))

    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_base)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10",
                          s=12, alpha=0.7)
    plt.colorbar(scatter, ticks=range(n_clusters), label="簇")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("GMM 聚类结果 PCA 投影")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pca_projection_v5.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  已保存 -> {path}")

    # t-SNE（样本太少时不稳定，跳过）
    if len(df) < 50:
        print("  [提示] 样本量 < 50，跳过 t-SNE（小样本不稳定）")
        return
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, max(5, len(df) // 5)))
    X_tsne = tsne.fit_transform(X_base)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10",
                          s=12, alpha=0.7)
    plt.colorbar(scatter, ticks=range(n_clusters), label="簇")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("GMM 聚类结果 t-SNE 投影")
    path = out_dir / "tsne_projection_v5.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  已保存 -> {path}")


# ======================== 稳健性检验（论文 5.3.1） ========================

def run_robustness_checks(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, label_names: list[str],
                          n_components: int, out_dir: Path):
    """
    P1 稳健性：
      1. K 敏感性：K∈{3,4,5} 各跑一次 5 折 CV，对比 Macro F1 / AUC
      2. Bootstrap 重采样：全量 GMM 簇占比的均值与 95% 区间
    """
    print("\n" + "=" * 60)
    print("[稳健性] P1 检验")
    print("=" * 60)

    # ---- K 敏感性 ----
    print("\n--- K 敏感性（K=3,4,5 的 5 折 CV 对比） ---")
    results = []
    for k in [3, 4, 5]:
        aucs = cross_validate_classifier_gmm(X, y, label_names, n_components=k)
        vals = [v for v in aucs.get("chemotherapy", []) if not np.isnan(v)]
        macro = vals[0] if vals else np.nan
        results.append({"K": k, "chemotherapy AUC": round(macro, 4) if not np.isnan(macro) else np.nan})
    sens_table = pd.DataFrame(results)
    print(sens_table.to_string())
    out_dir.mkdir(parents=True, exist_ok=True)
    sens_table.to_csv(out_dir / "robust_k_sensitivity_v5.csv", encoding="utf-8-sig")

    # ---- Bootstrap 重采样 ----
    print("\n--- Bootstrap 重采样稳定性（簇占比 95% 区间） ---")
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X_raw = df[feature_cols].copy()
    for col in WEIGHT_COLS:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col] * WEIGHT_MULTIPLIER
    pre = build_preprocessor(
        filter_existing_columns(CATEGORICAL_COLS, X_raw),
        filter_existing_columns(NUMERICAL_COLS, X_raw),
        filter_existing_columns(COMORBIDITY_COLS, X_raw) + filter_existing_columns(OTHER_BINARY_COLS, X_raw),
    )
    X_base = pre.fit_transform(X_raw)

    rng = np.random.default_rng(RANDOM_STATE)
    n_boot = 200
    n = len(X_base)
    fracs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gmm = GaussianMixture(n_components=n_components, covariance_type=GMM_COVARIANCE_TYPE,
                              random_state=RANDOM_STATE, n_init=GMM_N_INIT, reg_covar=GMM_REG_COVAR)
        try:
            gmm.fit(X_base[idx])
            lab = gmm.predict(X_base[idx])
        except Exception:
            continue
        counts = np.bincount(lab, minlength=n_components)
        fracs.append(counts / n)

    if fracs:
        fracs = np.array(fracs)
        boot_rows = []
        for c in range(n_components):
            col = fracs[:, c]
            boot_rows.append({
                "簇": c,
                "占比均值": round(col.mean(), 4),
                "占比95%下限": round(np.percentile(col, 2.5), 4),
                "占比95%上限": round(np.percentile(col, 97.5), 4),
            })
        boot_table = pd.DataFrame(boot_rows)
        print(boot_table.to_string())
        boot_table.to_csv(out_dir / "robust_bootstrap_v5.csv", encoding="utf-8-sig")
    else:
        print("  [警告] Bootstrap 全部拟合失败")


# ======================== 验证入口 ========================

def run_validation(figs_dir: Path):
    """
    独立验证：从 v5 产物重载模型并输出论文 5.3 全部统计结果。
    """
    print("\n[5/5] 统计验证（论文 5.3）...")
    if not GMM_PATH.exists() or not PREPROCESSOR_PATH.exists() or not DF_PATH.exists():
        print("[错误] 缺少 v5 模型文件，请先运行 --mode train 或 --mode all")
        return

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    gmm = joblib.load(GMM_PATH)
    df = pd.read_pickle(DF_PATH)
    k_opt = gmm.n_components
    print(f"  加载 GMM (K={k_opt})、预处理器、患者 DataFrame ({len(df)} 条)")

    labels = get_soft_labels(preprocessor, df, gmm)
    label_names = ["radiotherapy", "chemotherapy", "targeted_therapy", "immunotherapy", "hormone_therapy"]

    build_cluster_profile_table(df, labels, figs_dir)
    anova_top_features(preprocessor, df, labels, n_top=10, out_dir=figs_dir)
    treatment_acceptance_analysis(df, labels, label_names, figs_dir)

    # 加载 BIC 元数据绘图（若存在）
    if CLUSTER_META_PATH.exists():
        with open(CLUSTER_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "ks" in meta and "bics" in meta:
            plot_bic_curve(meta["ks"], meta["bics"], figs_dir)
    plot_projection(preprocessor, df, labels, figs_dir)


# ======================== 主流程 ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GMM 版患者检索 + 治疗预测训练与论文 5.3 统计验证")
    parser.add_argument("--data", type=str, default=str(DEFAULT_INPUT_JSON),
                        help=f"患者 JSON 路径（默认 {DEFAULT_INPUT_JSON}）；传 smoke 用合成数据")
    parser.add_argument("--mode", choices=["train", "validate", "all"], default="all",
                        help="train=训练, validate=仅验证, all=训练+验证（默认）")
    parser.add_argument("--k-min", type=int, default=K_MIN, help="BIC 候选聚类数下界（默认 2）")
    parser.add_argument("--k-max", type=int, default=K_MAX, help="BIC 候选聚类数上界（默认 10）")
    parser.add_argument("--figs-dir", type=str, default=str(RESULTS_DIR), help="图表/统计表输出目录")
    parser.add_argument("--robust", action="store_true", help="开启 P1 稳健性检验（K 敏感性 + Bootstrap）")
    parser.add_argument("--smoke", action="store_true", help="等同 --data smoke（用合成数据冒烟）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figs_dir = Path(args.figs_dir)

    # ---- 数据准备 ----
    if args.smoke or args.data.lower() == "smoke":
        print("[冒烟模式] 使用内置合成数据...")
        df = generate_synthetic_data(n=300)
    else:
        df = load_data(Path(args.data))

    best_k = None
    preprocessor = None
    gmm = None

    if args.mode in ("train", "all"):
        # ---- 1. 构建并保存检索系统 ----
        df, X_final, gmm, knn, preprocessor, best_k, cluster_meta = build_retrieval_system_gmm(
            df, k_min=args.k_min, k_max=args.k_max
        )
        save_retrieval_artifacts_v5(preprocessor, gmm, X_final, knn, df, cluster_meta)

        # ---- 2. 准备分类数据 ----
        X_cls, y_cls, label_names = prepare_label_data(df)

        # ---- 3. 交叉验证（获得无偏性能估计） ----
        cross_validate_classifier_gmm(X_cls, y_cls, label_names, k_min=args.k_min, k_max=args.k_max)

        # ---- 4. 全量训练最终模型（复用检索系统的 preprocessor + GMM） ----
        train_final_classifier_gmm(X_cls, y_cls, label_names, preprocessor, gmm)

        # ---- 5. 稳健性检验（可选） ----
        if args.robust:
            run_robustness_checks(df, X_cls, y_cls, label_names, best_k, figs_dir)

    if args.mode in ("validate", "all"):
        run_validation(figs_dir)

    print("\n" + "=" * 60)
    print("✅ v5 (GMM) 全流程完成！")
    print("=" * 60)
    for p in [PREPROCESSOR_PATH, GMM_PATH, FEATURES_PATH, KNN_PATH, IDS_PATH, DF_PATH,
              CLUSTER_META_PATH, CLASSIFIER_PATH]:
        print(f"   {p}")
    print(f"统计验证输出目录: {figs_dir}")


if __name__ == "__main__":
    main()
