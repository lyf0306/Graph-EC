"""
合并症筛查 Agent（Comorbidity Screening Agent）

核心流程：
  1. 【覆盖度分析】Agent 审查相似患者是否已满足当前患者的合并症筛查需求
  2. 【Skill 查参考】对未被相似患者充分覆盖的合并症，调用 reference_loader 检索参考文档
  3. 【综合输出】整合覆盖分析 + 参考文档检索结果，生成结构化的筛查报告

该设计将原本的单次 LLM 调用 + 硬编码 NCCN 词典，
升级为可判断 → 可查参考的 Agent 模式。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prompts import (
    COVERAGE_ANALYSIS_SYSTEM_PROMPT,
    COVERAGE_ANALYSIS_USER_TEMPLATE,
    REFERENCE_SKILL_SYSTEM_PROMPT,
    REFERENCE_SKILL_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from .reference_loader import ComorbidityReferenceLoader

logger = logging.getLogger(__name__)

# 默认参考文档路径（单文件模式）
DEFAULT_REFERENCE_FILE = Path(__file__).resolve().parent.parent.parent / "references" / "NCCN_2026v1.md"


class ComorbidityScreeningSkill:
    """
    合并症筛查 Skill
    将单一的 LLM 合并症判断升级为 Agent 模式：
    先审查相似患者的覆盖度 → 不满足则查参考文档 → 输出结构化结果
    """

    def __init__(
        self,
        llm_client: Any,
        reference_path: Optional[Path] = None,
        model_name: str = "deepseek-chat",
    ):
        """
        Args:
            llm_client: OpenAI-compatible async client (AsyncOpenAI instance)
            reference_path: 参考文档路径（单个 .md 文件），默认使用内置 NCCN_2026v1.md
            model_name: LLM 模型名
        """
        self.client = llm_client
        self.model = model_name
        self.reference_path = reference_path or DEFAULT_REFERENCE_FILE
        self.reference_loader = ComorbidityReferenceLoader(self.reference_path)

    async def screen(
        self,
        patient_dict: Dict,
        patient_profile_md: str,
        similar_patients_df: Any,
        mdt_overrides_str: str = "",
    ) -> Dict:
        """
        主入口：执行完整的合并症筛查流程

        Args:
            patient_dict: 患者结构化特征字典
            patient_profile_md: 患者 markdown 格式画像
            similar_patients_df: 相似患者 DataFrame（含合并症特征和 Y_text）
            mdt_overrides_str: MDT 降级考量文本

        Returns:
            {
                "filtered_cases": [...],       # 质控后的病例列表
                "safety_warnings": [...],       # 用药安全警告
                "deescalation_advice": [...],   # 方案降级建议
                "monitoring_recommendations": [...],  # 监测建议
                "coverage_summary": {...},      # 覆盖度摘要
                "reference_log": [...],         # 查了哪些参考文档
            }
        """
        logger.info("[ComorbidityScreeningSkill] 开始合并症筛查...")

        # ========== Step 1: 提取患者合并症特征 ==========
        comorbidity_features = self._extract_comorbidity_features(patient_dict)
        if not comorbidity_features:
            logger.warning("患者无明显合并症特征，跳过筛查")
            return self._empty_result()

        # ========== Step 2: 格式化相似患者数据 ==========
        similar_cases_text = self._format_similar_cases(similar_patients_df)

        # ========== Step 3: Agent 分析覆盖度 ==========
        logger.info("[Phase 1] Agent 分析相似患者合并症覆盖度...")
        coverage_result = await self._analyze_coverage(
            patient_profile_md=patient_profile_md,
            comorbidity_features=comorbidity_features,
            similar_cases_text=similar_cases_text,
            n_cases=len(similar_patients_df) if similar_patients_df is not None else 0,
        )

        comorbidities_needing_ref = coverage_result.get("comorbidities_needing_reference", [])

        # ========== Step 4: Skill 查参考 ==========
        reference_results = {}
        if comorbidities_needing_ref:
            logger.info(
                f"[Phase 2] Agent 判定 {len(comorbidities_needing_ref)} 项合并症需要查参考文档: "
                f"{comorbidities_needing_ref}"
            )
            reference_results = await self._lookup_references(
                comorbidities_needing_ref=comorbidities_needing_ref,
                patient_dict=patient_dict,
            )
        else:
            logger.info("[Phase 2] 所有合并症均已被相似病例充分覆盖，无需查参考文档")

        # ========== Step 5: 综合生成最终输出 ==========
        logger.info("[Phase 3] 综合生成筛查报告...")
        final_result = await self._synthesize_result(
            patient_profile_md=patient_profile_md,
            coverage_analysis=json.dumps(coverage_result, ensure_ascii=False, indent=2),
            reference_results=json.dumps(reference_results, ensure_ascii=False, indent=2),
            similar_cases_text=similar_cases_text,
            mdt_overrides_str=mdt_overrides_str,
        )

        # 补充元信息
        final_result["coverage_summary"] = {
            "total_comorbidities": len(coverage_result.get("analysis", [])),
            "covered": sum(
                1 for a in coverage_result.get("analysis", []) if a.get("status") == "covered"
            ),
            "partial": sum(
                1 for a in coverage_result.get("analysis", []) if a.get("status") == "partial"
            ),
            "uncovered": sum(
                1 for a in coverage_result.get("analysis", []) if a.get("status") == "uncovered"
            ),
            "reference_lookup_count": len(comorbidities_needing_ref),
        }
        final_result["reference_log"] = [
            {"comorbidity": c, "source": "similar_patients"}
            if c not in comorbidities_needing_ref
            else {"comorbidity": c, "source": "reference_docs"}
            for c in self._list_patient_comorbidities(patient_dict)
        ]

        logger.info(
            f"[ComorbidityScreeningSkill] 筛查完成: "
            f"{final_result['coverage_summary']['covered']} covered, "
            f"{final_result['coverage_summary']['reference_lookup_count']} via reference"
        )
        return final_result

    # ======================== 内部方法 ========================

    def _extract_comorbidity_features(self, patient_dict: Dict) -> str:
        """从患者特征字典中提取合并症相关的特征，格式化为文本"""
        comorbidity_keys = [
            "glycemic_status", "hypertension", "bmi_status", "hyperlipidemia",
            "anemia", "hepatic_viral", "hepatic_dysfunction", "major_cv_risk",
            "hpv_status",
        ]
        labels = {
            "glycemic_status": {0: "正常", 1: "糖耐量异常", 2: "糖尿病"},
            "bmi_status": {0: "正常", 1: "超重", 2: "肥胖"},
            "anemia": {0: "无", 1: "轻度", 2: "中重度"},
        }
        binary_label = {0: "无", 1: "有"}

        parts = []
        for key in comorbidity_keys:
            val = patient_dict.get(key, 0)
            if val and val != "unknown" and val != 0:
                if key in labels:
                    label = labels[key].get(int(val), str(val))
                else:
                    label = binary_label.get(int(val) if isinstance(val, (int, float)) else 0, str(val))
                parts.append(f"{key}: {label}")

        # 补充来自画像的文本化合并症描述
        text_comorbidities = patient_dict.get("comorbidities", "")
        if text_comorbidities and text_comorbidities != "无":
            parts.append(f"文本描述: {text_comorbidities}")

        return "\n".join(parts) if parts else "无明显合并症"

    def _list_patient_comorbidities(self, patient_dict: Dict) -> List[str]:
        """列出患者实际存在的合并症列表"""
        comorbidity_keys = [
            "glycemic_status", "hypertension", "bmi_status", "hyperlipidemia",
            "anemia", "hepatic_viral", "hepatic_dysfunction", "major_cv_risk",
            "hpv_status",
        ]
        result = []
        for key in comorbidity_keys:
            val = patient_dict.get(key, 0)
            if val and val != "unknown" and val != 0:
                result.append(key)
        return result

    def _format_similar_cases(self, df) -> str:
        """将相似患者的 DataFrame 格式化为 Agent 可读的文本"""
        if df is None or len(df) == 0:
            return "无可用相似病例。"

        from utils.trans_format import format_patient_desc

        parts = []
        for i, (idx, row) in enumerate(df.iterrows()):
            order = ["最相似", "次相似", "第三相似"][i] if i < 3 else f"第{i+1}相似"
            parts.append(f"### 病例 {idx}（{order}）")
            parts.append(format_patient_desc(row))
            y_text = row.get("Y_text", "")
            if y_text and y_text.strip() and y_text != "unknown":
                parts.append(f"历史治疗建议原文: {y_text}")
            parts.append("")
        return "\n".join(parts)

    async def _call_llm_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> Dict:
        """调用 LLM 并解析 JSON 响应"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            # 健壮的 JSON 提取
            import re

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 响应 JSON 解析失败: {e}")
            logger.debug(f"原始响应: {raw}")
            return {}
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {}

    async def _analyze_coverage(
        self,
        patient_profile_md: str,
        comorbidity_features: str,
        similar_cases_text: str,
        n_cases: int,
    ) -> Dict:
        """Phase 1: Agent 分析相似患者的合并症覆盖度"""
        user_prompt = COVERAGE_ANALYSIS_USER_TEMPLATE.format(
            patient_profile=patient_profile_md,
            comorbidity_features=comorbidity_features,
            similar_cases=similar_cases_text,
            n_cases=n_cases,
        )
        return await self._call_llm_json(
            system_prompt=COVERAGE_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    async def _lookup_references(
        self,
        comorbidities_needing_ref: List[str],
        patient_dict: Dict,
    ) -> Dict[str, Dict]:
        """
        Phase 2: Skill 查参考
        对需要查参考的合并症，分别检索参考文档并用 LLM 生成结构化指导
        """
        results = {}
        other_comorbidities = self._extract_comorbidity_features(patient_dict)

        for comorbidity in comorbidities_needing_ref:
            # 先从参考文档中获取原始内容
            doc_content = self.reference_loader.get_relevant_context(comorbidity)

            if not doc_content:
                # 尝试别名匹配
                mapped = ComorbidityReferenceLoader.TOPIC_MAP.get(comorbidity)
                if mapped:
                    doc_content = self.reference_loader.get_relevant_context(mapped)
                if not doc_content:
                    logger.warning(f"未找到合并症 '{comorbidity}' 的参考文档")
                    results[comorbidity] = {"error": "no_reference_doc_found"}
                    continue

            # 用 LLM 从参考文档中提取结构化指导
            user_prompt = REFERENCE_SKILL_USER_TEMPLATE.format(
                comorbidity_name=comorbidity,
                reference_content=doc_content,
                other_comorbidities=other_comorbidities,
            )
            structured_guidance = await self._call_llm_json(
                system_prompt=REFERENCE_SKILL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            structured_guidance["_raw_doc_length"] = len(doc_content)
            results[comorbidity] = structured_guidance

        return results

    async def _synthesize_result(
        self,
        patient_profile_md: str,
        coverage_analysis: str,
        reference_results: str,
        similar_cases_text: str,
        mdt_overrides_str: str,
    ) -> Dict:
        """Phase 3: 综合所有信息生成最终筛查报告"""
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            patient_profile=patient_profile_md,
            coverage_analysis=coverage_analysis,
            reference_results=reference_results,
            similar_cases=similar_cases_text,
            mdt_overrides=mdt_overrides_str,
        )
        result = await self._call_llm_json(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # 确保关键字段存在
        result.setdefault("filtered_cases", [])
        result.setdefault("safety_warnings", [])
        result.setdefault("deescalation_advice", [])
        result.setdefault("monitoring_recommendations", [])
        return result

    def _empty_result(self) -> Dict:
        """返回空的筛查结果"""
        return {
            "filtered_cases": [],
            "safety_warnings": [],
            "deescalation_advice": [],
            "monitoring_recommendations": [],
            "coverage_summary": {
                "total_comorbidities": 0,
                "covered": 0,
                "partial": 0,
                "uncovered": 0,
                "reference_lookup_count": 0,
            },
            "reference_log": [],
        }
