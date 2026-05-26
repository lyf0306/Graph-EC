#!/usr/bin/env python3
"""
端到端流水线整合脚本 (项目根目录执行)

目录结构要求:
    project_root/
    ├── run_full_pipeline.py   ← 本脚本
    ├── src/                   ← 所有 Python 模块
    │   ├── v4_llm_pipeline.py
    │   ├── v4_comorbidity_extractor.py
    │   ├── v4_merge_features.py
    │   └── v4_train_model.py
    ├── data/                  ← 输入输出数据
    ├── models/                ← 模型持久化
    └── logs/                  ← 日志文件

执行顺序:
    1. src/v4_llm_pipeline.py
    2. src/v4_comorbidity_extractor.py
    3. src/v4_merge_features.py
    4. src/v4_train_model.py
"""

import sys
import subprocess
from pathlib import Path

# 确定项目根目录（即本脚本所在目录）
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

def run_script(script_path: Path, description: str) -> None:
    """在 src 目录下执行指定脚本"""
    print("\n" + "=" * 70)
    print(f"▶ 步骤：{description}")
    print(f"▶ 执行脚本：src/{script_path.name}")
    print("=" * 70)

    if not script_path.exists():
        print(f"❌ 错误：脚本不存在 {script_path}")
        sys.exit(1)

    # 切换工作目录到 src/，保证脚本内部的相对路径计算正确
    result = subprocess.run(
        [sys.executable, str(script_path.name)],
        cwd=str(SRC_DIR),               # 在 src 目录下执行
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ 脚本 {script_path.name} 执行失败，返回码 {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ 脚本 {script_path.name} 执行成功。")

def main():
    print("🚀 启动端到端训练流水线")
    print(f"📁 项目根目录：{PROJECT_ROOT}")
    print(f"📁 脚本目录  ：{SRC_DIR}")

    # 1. LLM 结构化提取（含 ESGO 风险分层）
    run_script(
        SRC_DIR / "v4_llm_pipeline.py",
        "LLM 结构化提取 (生成 structured_output_v4.json，已含 ESGO 风险组)"
    )

    # 2. 合并症独立提取
    run_script(
        SRC_DIR / "v4_comorbidity_extractor.py",
        "合并症独立提取 (生成 comorbidity_output_v4.json)"
    )

    # 3. 合并主病特征与合并症
    run_script(
        SRC_DIR / "v4_merge_features.py",
        "特征合并 (生成 final_patient_vectors_v4.json)"
    )

    # 4. 训练模型
    run_script(
        SRC_DIR / "v4_train_model.py",
        "训练检索系统与分类器 (生成 models/ 目录下所有模型文件)"
    )

    print("\n" + "=" * 70)
    print("🎉 全流水线执行完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()