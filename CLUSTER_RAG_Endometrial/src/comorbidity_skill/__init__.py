"""
合并症筛查 Skill 包

将相似患者聚类后的合并症筛查从"单次 LLM 调用 + 硬编码 NCCN 词典"
升级为 Agent 模式：

  1. Agent 分析相似患者是否已满足当前患者的合并症筛查需求
  2. 对未被充分覆盖的合并症，调用 Skill 查询参考文档（NCCN markdown 文件）
  3. 综合输出结构化筛查报告

使用方式：
    from comorbidity_skill import ComorbidityScreeningSkill

    skill = ComorbidityScreeningSkill(llm_client=client)
    result = await skill.screen(
        patient_dict=new_patient_dict,
        patient_profile_md=patient_profile_md,
        similar_patients_df=retrieved_df,
        mdt_overrides_str=mdt_overrides_str,
    )
"""

from .agent import ComorbidityScreeningSkill
from .reference_loader import ComorbidityReferenceLoader, ReferenceDoc

__all__ = [
    "ComorbidityScreeningSkill",
    "ComorbidityReferenceLoader",
    "ReferenceDoc",
]
