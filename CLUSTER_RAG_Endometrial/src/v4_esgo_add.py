# 后处理脚本

import json
from pathlib import Path
from v4_esgo_decision_tree import classify_esgo_risk, recommend_adjuvant_therapy

# 获取当前脚本所在目录，然后上升到父目录，再进入 data 文件夹
base_dir = Path(__file__).parent.parent  # 脚本的上级目录的再上级（即项目根目录）
data_dir = base_dir / "data"

input_file = data_dir / "structured_output_v4.json"
output_file = data_dir / "structured_output_v4_with_risk.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    X = item["X"]
    X["esgo_risk_group"] = classify_esgo_risk(X)
    X["esgo_recommendation"] = recommend_adjuvant_therapy(
        X["esgo_risk_group"],
        X.get("molecular_subtype", "unknown"),
        X.get("stage_2023", "unknown")
    )

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)