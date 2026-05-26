"""
v4_merge_features.py
按 id 合并主病特征（structured_output_v4.json）与合并症特征（comorbidity_output_v4.json），
生成最终患者特征向量 final_patient_vectors_v4.json
"""

import json
from pathlib import Path

# ======================== 配置 ========================
BASE_DIR = Path(__file__).parent.parent  # 假设脚本放在项目根目录的 scripts/ 或直接根目录
DATA_DIR = BASE_DIR / "data"

MAIN_FILE = DATA_DIR / "structured_output_v4_with_risk.json"   # 已包含 ESGO 风险组的主病数据
COMORBIDITY_FILE = DATA_DIR / "comorbidity_output_v4.json"     # 独立提取的合并症数据
OUTPUT_FILE = DATA_DIR / "final_patient_vectors_v4.json"       # 合并后的完整特征向量

# 合并症字段默认值（当某患者缺失合并症数据时使用）
DEFAULT_COMORBIDITY = {
    "glycemic_status": 0,
    "hypertension": 0,
    "bmi_status": 0,
    "hyperlipidemia": 0,
    "anemia": 0,
    "hepatic_viral": 0,
    "hepatic_dysfunction": 0,
    "major_cv_risk": 0
}


def load_main_data(file_path: Path) -> dict:
    """加载主病数据，返回以 id 为键的字典"""
    if not file_path.exists():
        raise FileNotFoundError(f"主病数据文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 假设 data 是一个列表，每个元素包含 "id" 和 "X" 等字段
    return {item["id"]: item for item in data}


def load_comorbidity_data(file_path: Path) -> dict:
    """加载合并症数据，返回以 id 为键的特征字典（仅合并症字段）"""
    if not file_path.exists():
        raise FileNotFoundError(f"合并症数据文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 假设 data 是一个列表，每个元素包含 "id" 和 "X"（内含8个合并症字段）
    comorbidity_dict = {}
    for item in data:
        pid = item.get("id")
        if pid is None:
            print(f"警告：合并症数据中存在缺失 id 的条目，已跳过")
            continue
        # 有的结构可能是 {"id": ..., "X": {...}}，有的可能直接是特征字典，这里做兼容
        if "X" in item:
            features = item["X"]
        else:
            features = {k: v for k, v in item.items() if k != "id"}
        comorbidity_dict[pid] = features
    return comorbidity_dict


def merge_features(main_dict: dict, como_dict: dict) -> list:
    """将合并症特征合并到主病数据的 X 字段中，返回合并后的列表"""
    merged_list = []
    missing_comorbidity_count = 0

    for pid, patient in main_dict.items():
        X = patient.get("X", {})
        if pid in como_dict:
            # 合并存在的合并症特征
            X.update(como_dict[pid])
        else:
            # 缺失合并症数据，使用默认值填充
            X.update(DEFAULT_COMORBIDITY)
            missing_comorbidity_count += 1
        patient["X"] = X
        merged_list.append(patient)

    print(f"合并完成：共 {len(merged_list)} 条记录，其中 {missing_comorbidity_count} 条缺失合并症数据，已用默认值填充。")
    return merged_list


def main():
    print("正在加载主病数据...")
    main_dict = load_main_data(MAIN_FILE)
    print(f"主病数据加载完毕，共 {len(main_dict)} 条记录。")

    print("正在加载合并症数据...")
    como_dict = load_comorbidity_data(COMORBIDITY_FILE)
    print(f"合并症数据加载完毕，共 {len(como_dict)} 条记录。")

    print("正在合并特征...")
    merged_data = merge_features(main_dict, como_dict)

    print(f"正在保存至 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("✅ 所有操作完成！")


if __name__ == "__main__":
    main()