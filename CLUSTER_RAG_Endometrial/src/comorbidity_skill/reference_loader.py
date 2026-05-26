"""
参考文档加载器（Reference Loader）
负责从 markdown 文件中加载和搜索合并症参考指南。
后续可扩展支持更多格式（PDF, DOCX 等）。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional


class ReferenceDoc:
    """单个参考文档"""
    def __init__(self, path: Path):
        self.path = path
        self.topic = path.stem  # 文件名（不含后缀）作为主题词
        self.content = path.read_text(encoding="utf-8")
        self._build_index()

    def _build_index(self):
        """构建简单的标题索引"""
        self.sections = {}
        current_section = "概述"
        for line in self.content.split("\n"):
            if line.startswith("## "):
                current_section = line.strip("# ").strip()
                self.sections[current_section] = []
            elif line.startswith("# "):
                current_section = line.strip("# ").strip()
                self.sections[current_section] = []
            else:
                self.sections.setdefault(current_section, []).append(line)

    def search(self, keyword: str) -> List[str]:
        """在文档中搜索关键词，返回匹配的段落"""
        results = []
        lines = self.content.split("\n")
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                snippet = "\n".join(lines[start:end])
                results.append(snippet)
        return results

    def get_section(self, section_name: str) -> Optional[str]:
        """获取指定章节的文本"""
        if section_name in self.sections:
            return "\n".join(self.sections[section_name])
        # 模糊匹配
        for key, lines in self.sections.items():
            if section_name.lower() in key.lower():
                return "\n".join(lines)
        return None

    def summary(self, max_len: int = 500) -> str:
        """返回文档前 max_len 字符作为摘要"""
        return self.content[:max_len] + ("..." if len(self.content) > max_len else "")


class ComorbidityReferenceLoader:
    """
    合并症参考文档加载器
    支持两种模式：
      1. 单文件模式：传入单个 .md 文件路径，按 ## 标题分节，通过关键词搜索返回相关章节
      2. 目录模式（旧）：从指定目录加载所有 markdown 文件（兼容保留）
    """

    # 内置合并症主题索引（文件名 → 标准字段名映射）
    TOPIC_MAP = {
        "hypertension": "hypertension",
        "glycemic_status": "glycemic_status",
        "major_cv_risk": "major_cv_risk",
        "hepatic_viral": "hepatic_viral",
        "hepatic_dysfunction": "hepatic_dysfunction",
        "anemia": "anemia",
        "hyperlipidemia": "hyperlipidemia",
        "renal_dysfunction": "renal_dysfunction",
        "autoimmune_disease": "autoimmune_disease",
        "thrombosis_risk": "thrombosis_risk",
        "bleeding_tendency": "bleeding_tendency",
        "peripheral_neuropathy": "peripheral_neuropathy",
        "hypersensitivity_history": "hypersensitivity_history",
        "pneumonitis_ild": "pneumonitis_ild",
        "thyroid_dysfunction": "thyroid_dysfunction",
        "wound_healing": "wound_healing",
        # 别名映射
        "diabetes": "glycemic_status",
        "cv_risk": "major_cv_risk",
        "cardiac": "major_cv_risk",
        "heart": "major_cv_risk",
        "stroke": "major_cv_risk",
        "liver": "hepatic_dysfunction",
        "hbv": "hepatic_viral",
        "hcv": "hepatic_viral",
        "hepatitis": "hepatic_viral",
        "kidney": "renal_dysfunction",
        "renal": "renal_dysfunction",
        "drug_allergy": "hypersensitivity_history",
        "ild": "pneumonitis_ild",
    }

    # 单文件模式下：合并症键名 → 搜索关键词（中英文）
    COMORBIDITY_KEYWORDS = {
        "glycemic_status": ["血糖", "糖尿病", "糖代谢", "glycemic", "胰岛素", "地塞米松"],
        "hypertension": ["高血压", "血压", "hypertension", "贝伐珠", "仑伐替尼"],
        "major_cv_risk": ["心脏", "cardiac", "卒中", "stroke", "心肌梗死", "心血管", "血栓", "栓塞", "cardiovascular", "他莫昔芬"],
        "hepatic_dysfunction": ["肝", "hepatic", "胆红素", "黄疸"],
        "hepatic_viral": ["肝炎", "乙肝", "HBV", "hepatitis"],
        "anemia": ["贫血", "anemia", "血液学"],
        "hyperlipidemia": ["高脂", "血脂", "lipid"],
        "thrombosis_risk": ["血栓", "栓塞", "thrombus", "他莫昔芬"],
        "bleeding_tendency": ["出血", "bleeding"],
        "hypersensitivity_history": ["过敏", "allergic", "hypersensitivity", "脱敏", "输注反应", "infusion"],
        "peripheral_neuropathy": ["神经病变", "neuropathy", "neurotoxicity", "感觉神经"],
        "pneumonitis_ild": ["间质性肺炎", "ILD", "pneumonitis"],
        "thyroid_dysfunction": ["甲状腺", "thyroid"],
        "autoimmune_disease": ["自身免疫", "autoimmune"],
        "renal_dysfunction": ["肾", "renal"],
        "wound_healing": ["伤口", "wound", "愈合", "碎瘤", "morcellation"],
        "bmi_status": ["肥胖", "BMI", "体重"],
        "hpv_status": ["HPV", "人乳头瘤", "宫颈", "免疫治疗", "hpv感染"],
    }

    def __init__(self, reference_path: Optional[Path] = None):
        self.reference_path = reference_path
        self.docs: Dict[str, ReferenceDoc] = {}
        self._single_file_mode = False
        self._sections: Dict[str, str] = {}  # heading -> content
        self._file_content: str = ""

        if reference_path and reference_path.exists():
            if reference_path.is_file():
                self._load_single_file(reference_path)
            elif reference_path.is_dir():
                self._load_all(reference_path)

    def _load_all(self, directory: Path):
        """加载目录下所有 .md 文件"""
        for md_file in sorted(directory.glob("*.md")):
            doc = ReferenceDoc(md_file)
            self.docs[doc.topic] = doc
            # 同时用中文名索引
            for line in md_file.read_text(encoding="utf-8").split("\n"):
                if line.startswith("# ") and "：" in line:
                    cn_name = line.strip("# ").split("（")[0].strip()
                    self.docs[cn_name] = doc
                break

    def _load_single_file(self, file_path: Path):
        """加载单个 markdown 文件，按 ## 标题解析为章节"""
        self._single_file_mode = True
        self._file_content = file_path.read_text(encoding="utf-8")

        current_heading = "概述"
        current_lines = []
        for line in self._file_content.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    self._sections[current_heading] = "\n".join(current_lines)
                current_heading = line.strip("# ").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            self._sections[current_heading] = "\n".join(current_lines)

    def _search_sections(self, comorbidity_key: str) -> str:
        """在单文件各章节中通过关键词搜索，返回相关章节内容"""
        # 收集所有搜索关键词
        keywords = list(self.COMORBIDITY_KEYWORDS.get(comorbidity_key, []))
        # 补充字段名本身
        keywords.append(comorbidity_key)
        # 尝试别名映射后补充关键词
        resolved = self.TOPIC_MAP.get(comorbidity_key)
        if resolved and resolved != comorbidity_key and resolved in self.COMORBIDITY_KEYWORDS:
            keywords.extend(self.COMORBIDITY_KEYWORDS[resolved])
        keywords = list(set(k.lower() for k in keywords if k))

        matched = []
        for heading, content in self._sections.items():
            content_lower = content.lower()
            if any(kw in content_lower for kw in keywords):
                matched.append(content)

        if matched:
            return "\n\n---\n\n".join(matched)
        # 无匹配时返回全文，让 LLM 自行判断
        return self._file_content

    def reload(self, reference_path: Optional[Path] = None):
        """重新加载参考文档"""
        if reference_path:
            self.reference_path = reference_path
        if self.reference_path and self.reference_path.exists():
            self.docs.clear()
            self._sections.clear()
            self._file_content = ""
            self._single_file_mode = False
            if self.reference_path.is_file():
                self._load_single_file(self.reference_path)
            elif self.reference_path.is_dir():
                self._load_all(self.reference_path)

    def get_doc(self, topic: str) -> Optional[ReferenceDoc]:
        """按主题获取参考文档"""
        # 直接匹配
        if topic in self.docs:
            return self.docs[topic]
        # 别名匹配
        resolved = self.TOPIC_MAP.get(topic)
        if resolved and resolved in self.docs:
            return self.docs[resolved]
        # 模糊匹配
        for key, doc in self.docs.items():
            if topic.lower() in key.lower():
                return doc
        return None

    def search(self, query: str) -> Dict[str, List[str]]:
        """在所有文档中搜索关键词，返回文档名 → 匹配段落列表"""
        if self._single_file_mode:
            query_lower = query.lower()
            results = {}
            for heading, content in self._sections.items():
                if query_lower in content.lower():
                    results[heading] = content
            if not results:
                results["全文"] = self._file_content
            return results

        results = {}
        query_lower = query.lower()
        for topic, doc in self.docs.items():
            # 跳过别名索引
            if topic in self.TOPIC_MAP and topic not in Path(self.reference_path or ".").iterdir():
                continue
            matches = doc.search(query_lower)
            if matches:
                results[topic] = matches
        return results

    def get_relevant_context(self, comorbidity_key: str) -> str:
        """获取某合并症对应的参考文档内容（作为 agent 上下文）"""
        if self._single_file_mode:
            return self._search_sections(comorbidity_key)
        # 目录模式（旧）
        doc = self.get_doc(comorbidity_key)
        if doc:
            return doc.content
        # 别名映射
        mapped = self.TOPIC_MAP.get(comorbidity_key)
        if mapped:
            doc = self.get_doc(mapped)
            if doc:
                return doc.content
        return ""

    def list_available_topics(self) -> List[str]:
        """列出所有可用的合并症主题"""
        if self._single_file_mode:
            return list(self._sections.keys())
        return [t for t in self.docs.keys() if t not in self.TOPIC_MAP or not isinstance(self.docs[t], ReferenceDoc)]

    @property
    def is_loaded(self) -> bool:
        return self._single_file_mode or len(self.docs) > 0
