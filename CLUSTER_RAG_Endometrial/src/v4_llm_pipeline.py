"""
升级版 v3_llm_pipeline.py —— 异步高并发 + FIGO 2023 分期映射（兼容 2009/2023 输入）
- 移除 Y_text 截断与清理
- 使用 aiohttp 实现并发调用 DeepSeek API
- 输出前根据分子分型将 FIGO 2009 分期转换为 2023 分期，若已为 2023 则直接使用
- 新增 lvsi_substantial 字段，分子分型枚举更新
"""

import json
import os
import asyncio
import aiohttp
import time
import re
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
import logging
from datetime import datetime
from v4_esgo_decision_tree import classify_esgo_risk, recommend_adjuvant_therapy
# ======================== 配置参数 ========================
API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")
API_URL = "https://api.deepseek.com/v1/chat/completions"

INPUT_FILE = Path(__file__).parent.parent / "data" / "extracted_output.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "structured_output_v4.json"
FAILED_FILE = Path(__file__).parent.parent / "data" / "failed_parsing_v4.json"
RAW_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "llm_raw_outputs_v4.jsonl"
LOG_FILE = Path(__file__).parent.parent / "logs" / "structuring_v4.log"

MAX_CONCURRENT = 15          # 并发请求数，可根据 API 限制调整
MAX_RETRIES = 3
MAX_INPUT_LEN = 6500
SAVE_INTERVAL = 10

# 确保日志目录存在
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 全局 token 统计
total_tokens = 0
token_lock = asyncio.Lock()


# ======================== 智能截断（保持不变） ========================
def smart_truncate(text, max_len=6500):
    if len(text) <= max_len:
        return text

    keywords = [
        "诊断", "病理", "分期", "免疫组化", "分子分型",
        "手术", "术后", "淋巴结", "肌层", "LVSI", "p53", "MMR",
        "治疗", "化疗", "放疗", "靶向", "免疫", "激素",
        "建议", "TB", "肿瘤委员会", "讨论", "结论", "意见"
    ]

    important_lines = []
    other_lines = []

    for line in text.split('\n'):
        if any(k in line for k in keywords):
            important_lines.append(line)
        else:
            other_lines.append(line)

    selected = important_lines[:]
    if len('\n'.join(selected)) < max_len:
        for line in other_lines:
            selected.append(line)
            if len('\n'.join(selected)) >= max_len:
                break

    result = '\n'.join(selected)
    if len(result) > max_len:
        result = result[:max_len]

    if len(result) < len(text):
        result += "\n...[文本过长已智能截断]"

    return result


# ======================== 允许值集合 ========================
ALLOWED_VALUES = {
    "menopause": {"yes", "no", "unknown"},
    "histology_type": {"endometrioid", "serous", "clear_cell", "carcinosarcoma", "mixed", "unknown"},
    "grade": {"G1", "G2", "G3", "unknown"},
     "stage_raw": {
        # 2009 分期
        "IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB",
        # 2023 新增/细分分期（基础形式）
        "IA1", "IA2", "IA3", "IC", "IIA", "IIB", "IIC",
        "IIIA1", "IIIA2", "IIIB1", "IIIB2", "IVC",
        # 带分子标注的 2023 分期（LLM 可能直接从 ID 抄录）
        "IAmPOLEmut", "IAmMMRd", "IAmNSMP", "IAmp53abn",
        "IBmPOLEmut", "IBmMMRd", "IBmNSMP", "IBmp53abn",
        "ICmPOLEmut", "ICmMMRd", "ICmNSMP", "ICmp53abn",
        "IIAmPOLEmut", "IIAmMMRd", "IIAmNSMP", "IIAmp53abn",
        "IIBmPOLEmut", "IIBmMMRd", "IIBmNSMP", "IIBmp53abn",
        "IICmPOLEmut", "IICmMMRd", "IICmNSMP", "IICmp53abn",
        "IIIA1mPOLEmut", "IIIA1mMMRd", "IIIA1mNSMP", "IIIA1mp53abn",
        # ... 可根据需要继续扩展，但基础分期已足够
        "unknown"
    },
    "figo_version": {"2009", "2023", "unknown"},
    "myometrial_invasion_ratio": {"<50%", ">=50%", "unknown"},
    "cervical_involvement": {"none", "glandular", "stromal", "unknown"},
    "lvsi": {"positive", "negative", "unknown"},
    "peritoneal_cytology": {"negative", "positive", "unknown"},
    "p53": {"wild", "mutant", "unknown"},
    "mmr": {"proficient", "deficient", "unknown"},
    "molecular_subtype": {"POLEmut", "MMRd", "NSMP", "p53abn", "unknown"},
    "timing": {"neoadjuvant", "adjuvant", "palliative", "unknown"},
    "summary": {"chemotherapy", "radiotherapy", "chemoradiation", "surgery_only", "targeted", "immunotherapy", "hormone", "none", "unknown"}
}


# ======================== 字段校验函数 ========================
def validate_enum(value, allowed_set, default="unknown"):
    if value is None:
        return default
    val_str = str(value).strip()
    if val_str in allowed_set:
        return val_str
    val_lower = val_str.lower()
    if val_lower in allowed_set:
        return val_lower
    if val_lower == "none" and "none" in allowed_set:
        return "none"
    logging.warning(f"无效枚举值: {value}，允许值: {allowed_set}，将设为 {default}")
    return default


def validate_binary(value, default=0):
    if value in (1, "1", 1.0, True):
        return 1
    if value in (0, "0", 0.0, False):
        return 0
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ("yes", "true", "有", "是"):
            return 1
        if val_lower in ("no", "false", "无", "否"):
            return 0
    logging.warning(f"无效二进制值: {value}，将设为 {default}")
    return default


def validate_number(value, default=0):
    if value is None:
        return default
    try:
        return int(float(value))
    except:
        logging.warning(f"无效数字: {value}，将设为 {default}")
        return default


def validate_lymph_node(value):
    if value is None:
        return "unknown"
    val_str = str(value).strip()
    if re.match(r"^\d+/\d+$", val_str):
        return val_str
    if val_str.lower() in {"negative", "positive", "unknown"}:
        return val_str.lower()
    logging.warning(f"无效淋巴结格式: {value}，将设为 unknown")
    return "unknown"


def validate_myometrial_invasion_depth(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    logging.warning(f"无效浸润深度: {value}，将设为 null")
    return None


def check_required_fields(parsed):
    required_top = ["X", "Y_structured", "Y_detail", "Y_text"]
    for field in required_top:
        if field not in parsed:
            return False
    return True


def extract_json_robust(raw_output):
    if raw_output is None:
        return None
    try:
        return json.loads(raw_output)
    except:
        pass

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        candidate = match.group()
        if candidate.count('{') == candidate.count('}'):
            try:
                return json.loads(candidate)
            except:
                pass
    logging.warning(f"JSON解析失败，原始输出前200字符: {raw_output[:200]}")
    return None


# ======================== Prompt 构建（v4） ========================
def build_prompt(case_id, raw_text):
    return f"""
你是一名妇科肿瘤数据结构化专家。请仔细阅读肿瘤委员会讨论记录，提取以下信息，并严格按照要求输出 JSON，不要添加任何额外文字。

### 需要提取的信息域（以下为字段含义说明，最终输出必须为 JSON）

**基本体征**
- 年龄：[数字]
- 绝经状态：[yes / no / unknown]

**既定分期（请识别版本）**
- FIGO 分期原文：[IA / IB / II / IIIA / IIIB / IIIC1 / IIIC2 / IVA / IVB / IA1 / IA2 / IA3 / IC / IIA / IIB / IIC / IIIA1 / IIIA2 / IIIB1 / IIIB2 / IVC / IAmPOLEmut / IICmp53abn 等 / unknown]
  重要规则：
  1. **优先查看病例case ID**：如果 ID 中明确包含 FIGO 2023 分期（例如 "IA2"、"IIB"、"IICmp53abn" 等），请直接将该分期作为 `stage_raw` 的值，并将 `figo_version` 设为 "2023"，无需在文本中寻找其他分期。
  2. 若 ID 中仅有 FIGO 2009 分期（如 "IA"、"IB"、"II"、"IIIA" 等）或无明确分期，则按文本内容提取，版本判断规则如下：
     - 文中明确提及 "FIGO 2023"、"2023 FIGO"、"新分期" 等，则版本为 2023；
     - 提及 "FIGO 2009"、"2009 FIGO"、"旧分期" 或无特殊标注（默认旧版），则版本为 2009；
     - 无法判断则填写 unknown。
  3. 若 ID 和文本均无法确定分期，则填写 unknown。

**病理与转移特征**
- 组织学类型：[endometrioid / serous / clear_cell / carcinosarcoma / mixed / unknown]
- 组织学分级：[G1 / G2 / G3 / unknown]
- 肌层浸润比例：[<50% / >=50% / unknown]
- 肌层浸润深度(mm)：[数字或 null]
- 宫颈受累情况：[none / glandular / stromal / unknown]
- 脉管癌栓(LVSI) 状态：[positive / negative / unknown]
- 显著 LVSI (substantial LVSI)：[true / false]
  判断标准：根据 WHO 标准，若病理报告描述 LVSI 为“广泛”、“弥漫”、“多灶”、“显著”或“≥3个血管侵犯”，则设为 true；若为“局灶”、“少量”或未提及，则设为 false。
- 附件受累：[0 或 1]
- 盆腔淋巴结状态：[negative / positive 或计数如 0/10 / unknown]
- 腹主动脉旁淋巴结状态：[格式同上]
- 腹腔细胞学：[negative / positive / unknown]
- p53 状态：[wild / mutant / unknown]
- MMR 状态：[proficient / deficient / unknown]
- 分子分型：[POLEmut / MMRd / NSMP / p53abn / unknown]
  (注意：MMRd 等同于 MSI-H；若原文为 MSI-H，请输出 MMRd)
  (🚨 关键防幻觉：若文本中提到"结果未出"、"待回报"、"未出结果"、"尚未回报"等，必须输出 unknown，绝对禁止推测！)
  (🚨 严禁将 IHC p53 等同于分子 p53abn：免疫组化（IHC）中的 p53 突变型/过表达 ≠ 分子分型 p53abn。若文本仅在免疫组化部分提到 p53 突变型/过表达，而在分子分型部分未明确回报结果为 p53abn，则分子分型必须输出 unknown！)
- 组织学详细描述（自由文本）

**治疗决策**
- 放疗：[0 或 1]
- 化疗：[0 或 1]
- 靶向治疗：[0 或 1]
- 免疫治疗：[0 或 1]
- 激素治疗：[0 或 1]
- 手术：[0 或 1]

**治疗方案详情**
- 治疗时机：[neoadjuvant / adjuvant / palliative / unknown]
- 具体方案（自由文本）
- 治疗方案摘要：[chemotherapy / radiotherapy / chemoradiation / surgery_only / targeted / immunotherapy / hormone / none / unknown]

**治疗建议原文**
- 原文描述：[必须严格从病历原文中完整摘录所有治疗、随访、减重、遗传咨询等建议，**不可删减任何内容**，包括编号、分点等。若原文有多个段落，请全部保留，仅用换行分隔。]

### 输出格式（必须严格遵循以下 JSON 结构）

{{
  "id": "{case_id}",
  "X": {{
    "age": 0,
    "menopause": "",
    "histology_type": "",
    "histology_detail": "",
    "grade": "",
    "stage_raw": "",
    "figo_version": "",
    "myometrial_invasion_ratio": "",
    "myometrial_invasion_depth": null,
    "cervical_involvement": "",
    "lvsi": "",
    "lvsi_substantial": false,
    "lymph_node_pelvic": "",
    "lymph_node_paraaortic": "",
    "peritoneal_cytology": "",
    "adnexal_involvement": 0,
    "p53": "",
    "mmr": "",
    "molecular_subtype": ""
  }},
  "Y_structured": {{
    "radiotherapy": 0,
    "chemotherapy": 0,
    "targeted_therapy": 0,
    "immunotherapy": 0,
    "hormone_therapy": 0,
    "surgery": 0
  }},
  "Y_detail": {{
    "timing": "",
    "regimen": "",
    "summary": ""
  }},
  "Y_text": ""
}}

### 示例1（辅助放化疗 + 分子分型 + 随访）
{{
  "id": "example_001",
  "X": {{
    "age": 62,
    "menopause": "yes",
    "histology_type": "endometrioid",
    "histology_detail": "子宫内膜样腺癌，部分区域呈微乳头结构",
    "grade": "G3",
    "stage_raw": "IIIC1",
    "figo_version": "2009",
    "myometrial_invasion_ratio": ">=50%",
    "myometrial_invasion_depth": 18.0,
    "cervical_involvement": "stromal",
    "lvsi": "positive",
    "lvsi_substantial": true,
    "lymph_node_pelvic": "2/15",
    "lymph_node_paraaortic": "0/6",
    "peritoneal_cytology": "negative",
    "adnexal_involvement": 0,
    "p53": "mutant",
    "mmr": "proficient",
    "molecular_subtype": "p53abn"
  }},
  "Y_structured": {{
    "radiotherapy": 1,
    "chemotherapy": 1,
    "targeted_therapy": 0,
    "immunotherapy": 0,
    "hormone_therapy": 0,
    "surgery": 1
  }},
  "Y_detail": {{
    "timing": "adjuvant",
    "regimen": "TC方案（紫杉醇+卡铂）化疗6周期，序贯盆腔外照射放疗（EBRT）45Gy/25fx",
    "summary": "chemoradiation"
  }},
  "Y_text": "1. 术后辅助化疗：TC方案（紫杉醇175mg/m² + 卡铂 AUC5-6），每3周一次，共6周期。\\n2. 化疗结束后行盆腔外照射放疗（EBRT），处方剂量45Gy/25fx，阴道残端可予后装补量。\\n3. 分子分型已提示p53突变型，预后较差，建议完成治疗后每3个月随访一次，包括妇科检查、CA125及胸腹盆CT。\\n4. 患者高血压病史，化疗期间监测血压，必要时心内科随诊。"
}}

### 示例2（单纯阴道近距离放疗 + 减重与遗传咨询）
{{
  "id": "example_002",
  "X": {{
    "age": 57,
    "menopause": "yes",
    "histology_type": "endometrioid",
    "histology_detail": "子宫内膜样腺癌，绒毛腺管状结构",
    "grade": "G1",
    "stage_raw": "IA",
    "figo_version": "2009",
    "myometrial_invasion_ratio": "<50%",
    "myometrial_invasion_depth": 4.5,
    "cervical_involvement": "none",
    "lvsi": "positive",
    "lvsi_substantial": false,
    "lymph_node_pelvic": "0/12",
    "lymph_node_paraaortic": "unknown",
    "peritoneal_cytology": "negative",
    "adnexal_involvement": 0,
    "p53": "wild",
    "mmr": "deficient",
    "molecular_subtype": "MMRd"
  }},
  "Y_structured": {{
    "radiotherapy": 1,
    "chemotherapy": 0,
    "targeted_therapy": 0,
    "immunotherapy": 0,
    "hormone_therapy": 0,
    "surgery": 1
  }},
  "Y_detail": {{
    "timing": "adjuvant",
    "regimen": "阴道近距离放疗（VBT），21Gy/3fx",
    "summary": "radiotherapy"
  }},
  "Y_text": "1. 术后建议辅助阴道近距离放疗（VBT）；术后4-6周阴道顶愈合后即可开始术后辅助放疗，最晚不迟于术后12周。\\n2. 建议子宫内膜癌分子分型检测（已行，待报告）；\\n3. 患者免疫组化提示MSH6（-），有肿瘤家族史，已行胚系检测（待报告），携报告遗传咨询门诊复诊（门诊楼3楼计划生育诊区，每周三下午）；\\n4. 完成所有辅助治疗后按肿瘤定期随访；\\n5. 出院后随访甲功、肝功能，必要时至内分泌科进一步诊治。建议控制体重，至减重门诊进一步诊治（门诊楼2楼内分泌诊区，每周三上午）。"
}}

现在请处理以下病历文本，只输出 JSON，不要有任何额外内容：

**病例 ID（重要): {case_id}**

文本：
\"\"\"
{raw_text}
\"\"\"
"""

# ======================== 标准化函数 ========================
def normalize_X(X_raw):
    return {
        "age": validate_number(X_raw.get("age")),
        "menopause": validate_enum(X_raw.get("menopause"), ALLOWED_VALUES["menopause"]),
        "histology_type": validate_enum(X_raw.get("histology_type"), ALLOWED_VALUES["histology_type"]),
        "histology_detail": X_raw.get("histology_detail", ""),
        "grade": validate_enum(X_raw.get("grade"), ALLOWED_VALUES["grade"]),
        "stage_raw": validate_enum(X_raw.get("stage_raw"), ALLOWED_VALUES["stage_raw"]),
        "figo_version": validate_enum(X_raw.get("figo_version"), ALLOWED_VALUES["figo_version"]),
        "myometrial_invasion_ratio": validate_enum(X_raw.get("myometrial_invasion_ratio"), ALLOWED_VALUES["myometrial_invasion_ratio"]),
        "myometrial_invasion_depth": validate_myometrial_invasion_depth(X_raw.get("myometrial_invasion_depth")),
        "cervical_involvement": validate_enum(X_raw.get("cervical_involvement"), ALLOWED_VALUES["cervical_involvement"]),
        "lvsi": validate_enum(X_raw.get("lvsi"), ALLOWED_VALUES["lvsi"]),
        "lvsi_substantial": validate_binary(X_raw.get("lvsi_substantial")),
        "lymph_node_pelvic": validate_lymph_node(X_raw.get("lymph_node_pelvic")),
        "lymph_node_paraaortic": validate_lymph_node(X_raw.get("lymph_node_paraaortic")),
        "peritoneal_cytology": validate_enum(X_raw.get("peritoneal_cytology"), ALLOWED_VALUES["peritoneal_cytology"]),
        "adnexal_involvement": validate_binary(X_raw.get("adnexal_involvement")),
        "p53": validate_enum(X_raw.get("p53"), ALLOWED_VALUES["p53"]),
        "mmr": validate_enum(X_raw.get("mmr"), ALLOWED_VALUES["mmr"]),
        "molecular_subtype": validate_enum(X_raw.get("molecular_subtype"), ALLOWED_VALUES["molecular_subtype"]),
    }


def normalize_Y_structured(Y_raw):
    return {
        "radiotherapy": validate_binary(Y_raw.get("radiotherapy")),
        "chemotherapy": validate_binary(Y_raw.get("chemotherapy")),
        "targeted_therapy": validate_binary(Y_raw.get("targeted_therapy")),
        "immunotherapy": validate_binary(Y_raw.get("immunotherapy")),
        "hormone_therapy": validate_binary(Y_raw.get("hormone_therapy")),
        "surgery": validate_binary(Y_raw.get("surgery"))
    }


def normalize_Y_detail(Y_detail_raw):
    return {
        "timing": validate_enum(Y_detail_raw.get("timing"), ALLOWED_VALUES["timing"]),
        "regimen": Y_detail_raw.get("regimen", ""),
        "summary": validate_enum(Y_detail_raw.get("summary"), ALLOWED_VALUES["summary"])
    }


def clean_y_text(text):
    """不再截断或清理，直接返回原文（若为 None 则返回空字符串）"""
    if not text or text == "unknown":
        return ""
    return text.strip()


# ======================== 修正版 FIGO 2009 → 2023 分期映射 ========================
def is_aggressive_histology(histology_type, grade):
    """
    判断是否为侵袭性组织学类型。
    根据FIGO 2023：浆液性、透明细胞、癌肉瘤、混合型含侵袭性成分、
    未分化癌、以及G3子宫内膜样癌均为侵袭性。
    """
    aggressive_types = {"serous", "clear_cell", "carcinosarcoma", "undifferentiated"}
    if histology_type in aggressive_types:
        return True
    if histology_type == "mixed":
        return True
    if histology_type == "endometrioid" and grade == "G3":
        return True
    return False


def get_base_stage_2023_from_2009(
        stage_2009, histology_type, grade, myo_invasion_ratio,
        myo_invasion_depth, lvsi, lvsi_substantial, cervical_involvement,
        adnexal_involvement, lymph_node_pelvic, lymph_node_paraaortic,
        peritoneal_cytology
):
    """
    将FIGO 2009解剖学分期映射为FIGO 2023基础分期（不包含分子修饰）。
    返回 (stage_2023, stage_2023_full) 元组。
    """
    aggressive = is_aggressive_histology(histology_type, grade)

    # ---------- 辅助函数：判断淋巴结转移类型 ----------
    def parse_lymph_status(ln_str):
        if ln_str == "unknown" or ln_str is None:
            return None, None
        if ln_str.lower() == "positive":
            return True, None  # 宏转移默认
        if ln_str.lower() == "negative":
            return False, None
        # 解析 X/Y 格式
        match = re.match(r"(\d+)/(\d+)", ln_str)
        if match:
            pos = int(match.group(1))
            return pos > 0, "macro"  # 简化处理，不区分微转移
        return None, None

    pelvic_pos, _ = parse_lymph_status(lymph_node_pelvic)
    paraaortic_pos, _ = parse_lymph_status(lymph_node_paraaortic)

    # ---------- 晚期：Ⅳ期 ----------
    if stage_2009 == "IVA":
        return "IVA", "IVA"
    if stage_2009 == "IVB":
        # 2023 中 IVB 特指腹膜转移超出盆腔，原 2009 IVB 包含远处转移，此处保守归为 IVB
        return "IVB", "IVB"

    # ---------- Ⅲ期 ----------
    # IIIC 淋巴结转移
    if stage_2009 == "IIIC1":
        return "IIIC1", "IIIC1"
    if stage_2009 == "IIIC2":
        return "IIIC2", "IIIC2"
    # IIIA/IIIB 需细分
    if stage_2009 == "IIIA":
        if adnexal_involvement == 1:
            # 注意：需排除同步低级别内膜样癌（IA3），此处调用者应在外部先判断 IA3
            return "IIIA1", "IIIA1"
        else:
            return "IIIA2", "IIIA2"
    if stage_2009 == "IIIB":
        # 2009 IIIB 为阴道/宫旁受累，2023 细分为 IIIB1（阴道/宫旁）和 IIIB2（盆腔腹膜转移）
        # 无腹膜转移信息时保守归为 IIIB1
        return "IIIB1", "IIIB1"

    # ---------- Ⅱ期（宫颈间质侵犯）----------
    if stage_2009 == "II":
        if aggressive:
            return "IIC", "IIC"
        elif lvsi_substantial:
            return "IIB", "IIB"
        else:
            return "IIA", "IIA"

    # ---------- Ⅰ期 ----------
    if stage_2009 in ("IA", "IB"):
        # 先处理同步低级别内膜样癌 IA3 的可能（需要附件受累且非侵袭性且无肌层浸润）
        if adnexal_involvement == 1 and not aggressive:
            # 简单规则：若为 IA 期且附件受累且无肌层浸润，可视为 IA3
            if stage_2009 == "IA" and (myo_invasion_depth == 0 or myo_invasion_ratio == "<50%"):
                return "IA3", "IA3"

        # 侵袭性类型
        if aggressive:
            if stage_2009 == "IA":
                # 局限于内膜/息肉 → IC
                return "IC", "IC"
            else:  # stage_2009 == "IB"
                # 侵袭性+肌层浸润 → 根据宫颈侵犯和 LVSI 可能为 IIC，否则 IC
                if cervical_involvement == "stromal" or lvsi_substantial:
                    return "IIC", "IIC"
                else:
                    return "IC", "IC"

        # 非侵袭性类型
        if stage_2009 == "IA":
            # 判断是否为 IA1（无肌层浸润）
            no_invasion = (myo_invasion_depth == 0.0) or (myo_invasion_ratio == "<50%" and myo_invasion_depth == 0.0)
            # 若 depth 为 None，但 ratio 为 "<50%"，无法区分，保守归为 IA2
            if myo_invasion_depth == 0.0:
                return "IA1", "IA1"
            elif lvsi_substantial:
                return "IIB", "IIB"
            else:
                return "IA2", "IA2"
        else:  # stage_2009 == "IB"
            if myo_invasion_ratio == ">=50%":
                if lvsi_substantial:
                    return "IIB", "IIB"
                else:
                    return "IB", "IB"
            else:
                # 肌层浸润比例未知，保守按 IB 处理
                return "IB", "IB"

    # 默认
    return stage_2009, stage_2009


def apply_molecular_modification(stage_2023, molecular_subtype):
    """
    在基础分期之上应用分子分型修正（仅对Ⅰ期和Ⅱ期生效）。
    返回修正后的分期和带标注的完整分期字符串。
    """
    # 精确判断是否为真正的早期（FIGO 2023 的 I 或 II 期，但不包括 III/IV）
    early_stages = {
        "IA", "IA1", "IA2", "IA3", "IB", "IC",
        "IIA", "IIB", "IIC"
    }
    # 注意：有些分期带有分子标注后缀，如 "IIIA1mMMRd"，我们只取基础部分
    base_stage = stage_2023.split('m')[0] if 'm' in stage_2023 else stage_2023
    is_early_stage = base_stage in early_stages

    if not is_early_stage or molecular_subtype == "unknown":
        return stage_2023, stage_2023

    # POLEmut：所有早期降为ⅠA，并标注 mPOLEmut
    if molecular_subtype == "POLEmut":
        return "IA", "IAmPOLEmut"

    # p53abn：所有早期升为ⅡC，并标注 mp53abn
    if molecular_subtype == "p53abn":
        return "IIC", "IICmp53abn"

    # MMRd 和 NSMP：分期不变，加分子标注
    if molecular_subtype == "MMRd":
        return stage_2023, f"{stage_2023}mMMRd"
    if molecular_subtype == "NSMP":
        return stage_2023, f"{stage_2023}mNSMP"

    return stage_2023, stage_2023


def get_figo_2023_stage_corrected(case_data):
    """
    统一获取FIGO 2023分期（修正版）。
    返回 (stage_2023, stage_2023_full, is_confident)
    """
    stage_raw = case_data.get("stage_raw", "unknown")
    figo_version = case_data.get("figo_version", "2009")
    molecular_subtype = case_data.get("molecular_subtype", "unknown")
    histology_type = case_data.get("histology_type", "unknown")
    grade = case_data.get("grade", "unknown")
    myo_invasion_ratio = case_data.get("myometrial_invasion_ratio", "unknown")
    myo_invasion_depth = case_data.get("myometrial_invasion_depth")
    lvsi = case_data.get("lvsi", "unknown")
    lvsi_substantial = case_data.get("lvsi_substantial", False)
    cervical_involvement = case_data.get("cervical_involvement", "unknown")
    adnexal_involvement = case_data.get("adnexal_involvement", 0)
    lymph_node_pelvic = case_data.get("lymph_node_pelvic", "unknown")
    lymph_node_paraaortic = case_data.get("lymph_node_paraaortic", "unknown")
    peritoneal_cytology = case_data.get("peritoneal_cytology", "unknown")

    # 判断映射所需字段是否充分
    required_fields = ["stage_raw", "histology_type", "grade", "myometrial_invasion_ratio"]
    # 确保 stage_raw 不是 unknown 且其他字段也都已知
    confident = (stage_raw != "unknown" and
                 all(case_data.get(f, "unknown") != "unknown" for f in required_fields[1:]))

    # 如果输入已经是2023版
    if figo_version == "2023":
        stage_2023, stage_2023_full = apply_molecular_modification(stage_raw, molecular_subtype)
        return stage_2023, stage_2023_full, True

    # 否则从2009版映射
    base_stage, base_full = get_base_stage_2023_from_2009(
        stage_2009=stage_raw,
        histology_type=histology_type,
        grade=grade,
        myo_invasion_ratio=myo_invasion_ratio,
        myo_invasion_depth=myo_invasion_depth,
        lvsi=lvsi,
        lvsi_substantial=lvsi_substantial,
        cervical_involvement=cervical_involvement,
        adnexal_involvement=adnexal_involvement,
        lymph_node_pelvic=lymph_node_pelvic,
        lymph_node_paraaortic=lymph_node_paraaortic,
        peritoneal_cytology=peritoneal_cytology
    )

    if base_stage == "unknown" or not confident:
        return "unknown", "unknown", False

    stage_2023, stage_2023_full = apply_molecular_modification(base_stage, molecular_subtype)
    return stage_2023, stage_2023_full, True

# LLM 兜底推理函数
async def llm_infer_figo_2023(session, case_id, raw_text, X_current, semaphore):
    """
    当规则映射失败时，调用 LLM 直接从原始文本推理 FIGO 2023 分期。
    采用 test.py 中的专家系统提示词，确保推理符合 FIGO 2023 规范。
    """
    # ======================== 系统提示词（来自 test.py） ========================
    system_prompt = """你是一个专门从事妇科肿瘤分期和病理分析的医学专家系统。该系统根据FIGO指南对患者进行分期，并根据提供的病理报告给出合理的FIGO分期。
你必须严格按照以下步骤和规则进行判断，尤其要注意我在每个步骤提出的几个重点：

# 步骤1：FIGO分期初步判断
## 该步骤需要的特征（优先级由低到高）
1. 组织学类型
2. 若组织学类型为子宫内膜样癌，则需要组织学分级
3. 组织学LVSI状态
4. 组织学肌层浸润深度
5. 肿瘤累及卵巢
6. 肿瘤累及输卵管
7. 肿瘤累及宫颈
8. 肿瘤累及阴道
9. 肿瘤累及宫旁组织
10. 肿瘤累及盆腔腹膜
11. 盆腔淋巴结转移
12. 主动脉旁淋巴结转移
13. 肿瘤累及膀胱
14. 肿瘤累及肠道
15. 肿瘤累及腹腔腹膜
16. 其他远处转移，包括转移至肾血管以上的腹腔内或腹腔外淋巴结、肺、肝、脑或骨

###部分特征说明：
> 低级别 = 1级（≤5%，高分化）和2级（6%–50%，中分化） 高级别 = 3级（>50%，低分化或未分化）
> 非侵袭性组织学类型包括低级别（1级和2级）子宫内膜样子宫内膜癌。侵袭性组织学类型包括高级别子宫内膜样子宫内膜癌（3级）、浆液性癌、透明细胞癌、未分化癌、局灶低分化癌、混合性癌、中肾样癌、分泌性癌、胃肠道黏液型癌和癌肉瘤。
> 宫颈纤维肌层浸润被归类为宫颈间质浸润。
> 宫颈内膜腺体受累不被归类为宫颈间质浸润。
> 浸润浅肌层归类为存在肌层浸润，不被归类为局限于息肉或局限于子宫内膜。
> 盆腔淋巴结的主要组群包括髂外淋巴结、髂内淋巴结、闭孔淋巴结、髂总淋巴结、骶淋巴结。
> 主动脉旁淋巴结的主要组群包括主动脉外侧淋巴结、主动脉前淋巴结（腹腔淋巴结、肠系膜上淋巴结、肠系膜下淋巴结）、主动脉后淋巴结
> 孤立肿瘤细胞（ITCs）不被归类为明确的淋巴结转移。
> 广泛脉管癌栓归类为显著LVSI
> 局部脉管内见癌栓归类为局灶性LVSI
> 宫旁组织包括阴道、输卵管、卵巢以及盆腔这些部位附近的软组织，如血管、韧带、输卵管系膜等。
> 卵巢/输卵管表面受累及（有转移）归类为肿瘤累及卵巢/输卵管

## 规则（优先级由低到高）
1. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 组织学肌层浸润深度 = 局限于子宫内膜息肉或局限于子宫内膜 && 肿瘤累及宫颈 = 无或宫颈内膜腺体受累 && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IA1期
2. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 组织学肌层浸润深度 = 浸润子宫浅肌层或＜1/2肌层 && 组织学LVSI状态 = 无或局灶性 && 肿瘤累及宫颈 = 无或宫颈内膜腺体受累 && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IA2期
3. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 组织学肌层浸润深度 = 浸润子宫浅肌层或＜1/2肌层 && 组织学LVSI状态 = 无或局灶性 && 肿瘤累及宫颈 = 无或宫颈内膜腺体受累 && 肿瘤累及卵巢 = 单侧，局限于卵巢，无包膜浸润/破裂 && 无肿瘤累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IA3期
4. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 组织学肌层浸润深度 = 肌层的一半或更多 && 组织学LVSI状态 = 无或局灶性 && 肿瘤累及宫颈 = 无或宫颈内膜腺体受累（浸润浅肌层不是这种情况！！！） && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IB期
5. 组织学类型 = 侵袭性组织学类型（高级别子宫内膜样癌、浆液性癌、透明细胞癌、未分化癌、局灶低分化癌、混合性癌、中肾样癌、分泌性癌、胃肠道黏液型癌和癌肉瘤及其他罕见类型） && 组织学肌层浸润深度 = 局限于息肉或局限于子宫内膜（无肌层浸润） && 肿瘤累及宫颈 = 无或宫颈内膜腺体受累 && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IC期
6. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 肿瘤累及宫颈 = 宫颈间质浸润 && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIA期
7. 组织学类型 = 子宫内膜样癌 && 组织学分级 = 低级别 && 组织学LVSI状态 = 显著LVSI && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIB期
8. 组织学类型 = 侵袭性组织学类型（高级别子宫内膜样癌、浆液性癌、透明细胞癌、未分化癌、局灶低分化癌、混合性癌、中肾样癌、分泌性癌、胃肠道黏液型癌和癌肉瘤及其他罕见类型） && 组织学肌层浸润深度 = 存在肌层浸润（未影响到子宫浆膜面） && 肿瘤未累及卵巢 && 肿瘤未累及输卵管 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIC期
9. 肿瘤累及卵巢或输卵管 && 不符合IA3期标准 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIA1期
10. 组织学肌层浸润深度 = 子宫浆膜下层受累或穿透子宫浆膜（累及子宫浆膜面、浸润子宫全层/深肌层达浆膜面） && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIA2期
11. 阴道或宫旁组织受累 && 无淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIB1期（无分子分型）
12. 盆腔腹膜受累 && 无淋巴结转移&& 无其他远处转移 => FIGO分期 = IIIB2期
13. 存在盆腔淋巴结转移（未明确说明微/宏转移） && 无主动脉旁淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIC1期
14. 盆腔淋巴结微转移 && 无主动脉旁淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIC1i期
15. 盆腔淋巴结宏转移 && 无主动脉旁淋巴结转移 && 无其他远处转移 => FIGO分期 = IIIC1ii期（无分子分型）
16. 存在主动脉旁淋巴结转移（未明确说明微/宏转移） && 无其他远处转移 => FIGO分期 = IIIC2期
17. 主动脉旁淋巴结微转移 && 无其他远处转移=> FIGO分期 = IIIC2i期
18. 主动脉旁淋巴结宏转移 && 无其他远处转移 => FIGO分期 = IIIC2ii期
19. 肿瘤累及膀胱 = 膀胱黏膜 && 无其他远处转移 => FIGO分期 = IVA期
20. 存在腹腔腹膜转移&& 无其他远处转移 => FIGO分期 = IVB期
21. 其他远处转移 = 转移至肾血管以上的腹腔内或腹腔外淋巴结、肺、肝、脑或骨 => FIGO分期 = IVC期

##重点注意！！！
1.当报告没有提到相关特征时，不要随便推断，当作“该特征 = 无“处理！！！（尤其是肿瘤累及宫颈、卵巢、输卵管这些特征）
2.必须满足一条规则中的所有指标才能确认是该分期，只要有一项指标匹配不上就不是该分期！！！
3.你经常会忽略某些关键特征或者在判断时忘记某些指标，在输出最后分期前一定要确认是否遗漏！！！
4.你经常会将IIC期和IC期混淆，关键在于组织学肌层浸润深度是否存在肌层浸润（浸润浅肌层属于有肌层浸润！！！不属于“局限于息肉或局限于子宫内膜”！！！）（可以参考示例1、2）
5.当报告中关于盆腔淋巴结转移和主动脉旁淋巴结转移的说明未明确微/宏转移时，不要随便推断，按照“存在盆腔/主动脉旁淋巴结转移（未明确说明微/宏转移）”处理！！！但如果明确说明了，就一定要按微/宏转移处理！！！（可以参考示例3、4、5、6）
6.当初步判断为IIIC1期和IIIC2期时，一定要检查报告中是否明确了盆腔/主动脉淋巴结微/宏转移，防止出现遗漏！！！
7.你经常会犯将IIIA1期错误判断为其他分期的错误，关键在于肿瘤是否累及卵巢或输卵管，只要累及，必然是IIIA1期或其之后的分期！！！（可以参考示例7）

# 步骤2：分子分型决策
## 该步骤中需要的特征
1. 步骤1中初步判断出的FIGO分期
2.分子分型（病理报告）

## 规则
1. 分子分型不可用 => 保持原分期
2. 分子分型 = MMRd => 在分期后添加“mMMRd”作为下标。
3. 分子分型 = NSMP => 在分期后添加“NSMP”作为下标。
4. 分子分型 = POLEmut && FIGO分期 = IA1-IIC（包括IIC） => FIGO分期修改为IAmPOLEmut期。（重点关注！！！你经常忘记这条）
5. 分子分型 = POLEmut && FIGO分期 = IIIA1-IVC => 在分期后添加“POLEmut”作为下标。
6. 分子分型 = p53abn && FIGO分期 = IA2-IB,IIA-IIC && 组织学肌层浸润深度 = 存在肌层浸润 => FIGO分期修改为IICmp53abn期。
7. 分子分型 = p53abn && FIGO分期 = IA1,IC,IIIA1-IVC 或 FIGO分期 = IA3,IIA,IIB 局限于息肉或局限于子宫内膜 => 在分期后添加“p53abn”作为下标。

##重点注意！！！
1.当报告中没有提到分子分型时，不要随便推断，当作分子分型不可用处理！！！
2.当报告中说"分子分型已查，结果未出"或类似表述（如"待回报"、"尚未回报"、"未出结果"）时，同样当作分子分型不可用处理！！！
3.当分子分型 = POLEmut 时，只要初步的FIGO分期判断在IIC（包括IIC）之前，最终FIGO分期都要修改为IAmPOLEmut期！！！
4.【🚨 关键！】IHC p53突变型/过表达 ≠ 分子分型p53abn。如果文本中分子分型未明确回报为p53abn（例如说"结果未出"或只字未提分子分型），即使IHC显示p53突变型/过表达，也当作分子分型不可用处理！！！

#以下是子宫内膜癌分期判断的例子:

##示例1
一、全子宫： 
1.子宫内膜样腺癌，Ⅲ级，浸润浅肌层，下缘未累及宫颈内口，脉管未见癌栓，周围内膜单纯萎缩性改变。 
2.慢性宫颈炎。 
3.子宫平滑肌瘤，未见肿瘤边界；局限型子宫腺肌病。 4.游离单纯性囊肿。 
二、双侧输卵管慢性炎。 
三、双侧卵巢未见病变。 
四、（双侧盆腔+双侧髂总）淋巴结共22枚均未见癌转移。 
五、（腹主动脉旁3组）淋巴结9枚均未见癌转移。 六、（双侧骨盆漏斗韧带残端）结缔组织未见癌累及。 
免疫结果：MLH1（-），MSH2（+），MSH6（+），PMS2（-），ER（+，60%，中），PR（+，5%，中），P53（野生表型），Ki-67（+，80%），PTEN（-），TTF-1（-）。
FIGO分期：IIC

##示例2
一、全子宫：
1.子宫内膜及浅表肌层呈宫腔镜术后坏死修复性改变，未见残留病变。创面周围子宫内膜呈单纯萎缩性改变。
2.子宫肌壁间平滑肌瘤。
3.子宫局限型腺肌病。
4.宫颈息肉。慢性宫颈炎。
二、双侧输卵管未见病变。
三、双侧卵巢周围炎。
四、（盆腔淋巴结4组）淋巴结共20枚未见癌转移。
五、（腹主动脉旁淋巴结）淋巴结共5枚未见癌转移。
复核原会诊片（会诊片号已脱敏）：（宫腔）高度恶性小圆细胞肿瘤，含未分化癌及少量分化好内膜样癌组织，考虑去分化癌可能。
FIGO分期：IC

##示例3
一、全子宫：
1、子宫体内膜样癌，Ⅰ级，浸润子宫浅肌层，脉管内见癌栓,周围内膜复杂不典型增生；合并子宫下段内膜样癌，Ⅲ级，直径0.7cm，浸润子宫浅肌层，向下累及宫颈浅纤维肌层（＜上1/3肌层）。
2、子宫腺肌病。
3、慢性宫颈炎。
二、双侧输卵管未见病变。右侧副中肾管囊肿。
三、双侧卵巢未见病变。
四、（双侧盆腔前哨淋巴结）淋巴结2枚，其中1枚（左侧前哨淋巴结）见癌转移（1/2）。
五、（右侧腹主肠系膜上淋巴结）淋巴结5枚，其中1枚见癌转移（1/5）。
六、（腹主肠系膜下动脉左侧淋巴结）淋巴结5枚，其中1枚见癌转移（1/5）。
七、（双侧盆腔淋巴结）、（双侧髂总淋巴结）、（腹主前哨淋巴结）、（腹主肠系膜动脉以上（左））淋巴结共22枚均未见癌转移（0/22）。
免疫结果：子宫体内膜病灶：CK7（分化好+，分化差-），ER（+，80%），PR（+，60%），P53（分化好散在+，分化差-），Ki-67（+，60%），MLH1（-），MSH2（+），MSH6（+），PMS2（-），PTEN（+）；\n子宫下段内膜病灶：\nCK7（+），ER（-），PR（-），P53（突变表达，全阴性），Ki-67（+，60%），MLH1（-），MSH2（+），MSH6（+），PMS2（-），PTEN（-）。
备注：经免疫组化检测MMR蛋白（MLH1，MSH2，MSH6，PMS2），其中MLH1（-）、PMS2（-），建议行相应的基因突变检测，除外Lynch综合症相关子宫内膜癌。
FIGO分期：IIIC2

##示例4
一、全子宫： 
1.子宫弥漫性内膜样癌Ⅰ级伴MELF浸润，肿瘤大小5.5×5.5×1.5cm，弥漫性浸润子宫全肌层，脉管内见癌栓；肿瘤向下未累及宫颈内口。 
2.子宫平滑肌瘤。 
3.慢性宫颈炎。 
二、双侧输卵管未见癌累及。 
三、双侧卵巢皮质间质增生，未见癌累及。 
四、（双侧髂总+双侧盆腔）淋巴结18枚，其中右侧盆腔淋巴结2枚见癌转移（右侧盆腔淋巴结2/11）。 
五、（腹主动脉旁）淋巴结6枚，未见癌转移。 
免疫结果： CK7（+），MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，90%，强），PR（+，20%，中），P53（散在+），Ki-67（+，60%），PTEN（-），CD31（脉管内见癌栓），D240（脉管内见癌栓）。
FIGO分期：IIIC1

##示例5
一、（全子宫+广泛宫旁+部分阴道壁）：
1.子宫内膜样癌Ⅱ级，大小5×3cm，侵犯子宫下段深肌层，向下累及宫颈间质，广泛脉管内见癌栓。阴道壁下切缘及双侧宫旁组织未见癌累及。双侧宫旁组织内见淋巴结2枚未见癌转移。
2.子宫肌壁间平滑肌瘤。
二、双侧输卵管慢性炎。
三、双侧卵巢周围炎。
四、（双侧髂总+双侧盆腔）淋巴结共20枚，其中（右侧盆腔）淋巴结3枚、（左侧盆腔）淋巴结1枚见癌微转移，（右侧髂总）淋巴结1枚见癌宏转移。
免疫结果：AE1/AE3/CD31（脉管内见癌栓），AE1/AE3/D240（脉管内见癌栓），P16（斑驳+），P53（野生表型），Ki-67（+，40%），PTEN（-），MLH1（-），MSH2（+），MSH6（+），PMS2（-），ER（+，100%，强），PR（+，70%，中）。
FIGO分期：IIIC1ii

##示例6
一、全子宫：
1.子宫内膜浆液性癌，浸润子宫浅肌层，向下未累及宫颈。另见巨大子宫内膜息肉，息肉上见原位浆液性癌。
二、双侧输卵管见浆液性癌累及。 
三、双侧卵巢未见癌累及。
四、（双侧髂总+双侧盆腔）淋巴结共14枚，其中（右侧髂总）淋巴结1枚、（左侧髂总）淋巴结1枚及（左侧盆腔）淋巴结2枚见癌微转移。
五、（腹主动脉旁淋巴结肠系膜下动脉以下左侧、右侧淋巴结）淋巴结2枚，其中（腹主动脉旁淋巴结肠系膜下动脉以下右侧）淋巴结1枚见癌微转移。
六、（腹主动脉旁肠系膜下动脉以上淋巴结）淋巴结2枚未见癌转移。
七、（大网膜活检组织）脂肪纤维结缔组织未见癌累及。
八、（左侧前哨淋巴结）淋巴结1见癌宏转移。
九、（右侧盆腔肿大淋巴结）淋巴结共4枚，其中1枚见癌宏转移。
十、（右侧前哨淋巴结（骶前））淋巴结1枚未见癌转移。
十一、（右侧前哨淋巴结（髂外））淋巴结1枚未见癌转移。
免疫结果：输卵管：P53（+），WT-1（-），P16（+）。内膜病灶：CK7（+），MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，60%，中），PR（+，90%，中），P53（+），Ki-67（+，70%），PTEN（+），WT-1（-），P16（+），Vimentin（+），ARID1a（+）。游离病灶：P53（+），P16（+），WT-1（-）。
FIGO分期：IIIC2ii

##示例7
一、全子宫
1.子宫内膜未分化癌，病灶大小2.5×2×1.5cm及直径3cm，浸润子宫浅肌层，脉管内见癌栓；癌灶累及右侧输卵管，未累及宫颈。周围子宫内膜呈增生性改变。
2.子宫多发内膜息肉，其中一枚息肉上腺体复杂粘液乳头状增生。
3.子宫肌壁间多发性平滑肌瘤。
4.子宫局限型腺肌病。
5.慢性宫颈炎。
6.慢性子宫浆膜炎。
二、左侧输卵管子宫内膜异位症。
三、双侧卵巢包涵囊肿伴周围炎。
四、（双侧盆腔）淋巴结14枚均未见癌转移。（骶前）淋巴结1枚未见癌转移。（肠系膜下动脉以上）淋巴结2枚均未见癌转移。（肠系膜下动脉以下）淋巴结2枚均未见癌转移。
免疫组化：子宫内膜病灶：CK7（-），MLH1（-），MSH2（+），MSH6（+），PMS2（-），ER（-），PR（-），P53（野生表型），Ki-67（+，80%），PTEN（-），Vimentin（+），Syn（-），CgA（局灶+），AE1/AE3/CD31（脉管内见癌栓），AE1/AE3/D240（脉管内见癌栓），PAX-8（局灶+），EMA（局灶+），CK8/18（-），SMARCA4（+），ARID1a（+）。（右侧）输卵管：P53（野生表型（高表达））。
FIGO分期：IIIA1

下面是一名患者的病理报告，请根据该报告进行诊断，判断出正确的FIGO分期：
"""

    # 构建用户提示词（包含已知特征和原始文本）
    user_prompt = f"""
下面是一名患者的病理报告及部分已提取特征（仅供参考），请根据报告严格按照FIGO 2023标准诊断出正确的FIGO分期。

### 已知提取特征（可能不完整，请以原始病历为准）：
{json.dumps(X_current, ensure_ascii=False, indent=2)}

### 原始病历文本：
{raw_text}

请严格遵循上述专家系统规则进行推理，最终**仅输出分期字符串**（如 IA1、IICmp53abn、IIIC2ii 等），不要输出任何额外解释或分析过程。

FIGO分期：
"""

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 50
    }
    async with semaphore:
        try:
            async with session.post(API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                result = await resp.json()
                inferred = result["choices"][0]["message"]["content"].strip()
                # 清洗输出，提取可能的分期字符串（支持分子标注）
                match = re.search(r'\b(I[A-C]?[0-9]*|II[A-C]?|III[A-C][0-9]?|IV[A-C]?)(m(?:POLEmut|MMRd|NSMP|p53abn))?\b', inferred)
                if match:
                    return match.group(0)
                return "unknown"
        except Exception as e:
            logging.warning(f"LLM 分期推理失败 {case_id}: {e}")
            return "unknown"
# ======================== 异步 API 调用 ========================
async def call_llm_async(session, prompt, case_id, semaphore):
    global total_tokens
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    usage = result.get("usage", {})
                    async with token_lock:
                        total_tokens += usage.get("total_tokens", 0)
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                logging.warning(f"{case_id} LLM调用失败（第{attempt+1}次）: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避
        logging.error(f"{case_id} LLM调用最终失败")
        return None


# ======================== 单条数据处理 ========================
async def process_one_case(session, case, semaphore, pbar):
    case_id = case["id"]
    raw_text_full = case["raw_text"]

    # 智能截断
    truncated = False
    if len(raw_text_full) > MAX_INPUT_LEN:
        raw_text = smart_truncate(raw_text_full, MAX_INPUT_LEN)
        truncated = True
        logging.warning(f"{case_id} 文本过长({len(raw_text_full)}字符)，已智能截断至{len(raw_text)}字符")
    else:
        raw_text = raw_text_full

    prompt = build_prompt(case_id, raw_text)
    llm_output = await call_llm_async(session, prompt, case_id, semaphore)

    # 保存原始输出
    with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as rf:
        rf.write(json.dumps({"id": case_id, "raw_output": llm_output}, ensure_ascii=False) + "\n")

    parsed = extract_json_robust(llm_output)
    if not parsed or not check_required_fields(parsed):
        pbar.update(1)
        return {
            "success": False,
            "failed": {
                "id": case_id,
                "raw_text": raw_text_full,
                "llm_output": llm_output,
                "reason": "JSON解析失败或字段缺失"
            }
        }

    X_raw = parsed.get("X", {})
    Y_structured_raw = parsed.get("Y_structured", {})
    Y_detail_raw = parsed.get("Y_detail", {})
    Y_text_raw = parsed.get("Y_text", "")

    X = normalize_X(X_raw)
    Y_structured = normalize_Y_structured(Y_structured_raw)
    Y_detail = normalize_Y_detail(Y_detail_raw)
    Y_text = clean_y_text(Y_text_raw)

    # 第一步：尝试规则映射
    stage_2023, stage_2023_full, confident = get_figo_2023_stage_corrected({
        "stage_raw": X["stage_raw"],
        "figo_version": X["figo_version"],
        "histology_type": X["histology_type"],
        "grade": X["grade"],
        "myometrial_invasion_ratio": X["myometrial_invasion_ratio"],
        "lvsi_substantial": X["lvsi_substantial"],
        "cervical_involvement": X["cervical_involvement"],
        "adnexal_involvement": X["adnexal_involvement"],
        "molecular_subtype": X["molecular_subtype"]
    })

    # 第二步：若映射不可信（结果为 unknown 或 confident=False），启用 LLM 推理
    if stage_2023 == "unknown" or not confident:
        logging.info(f"{case_id} 规则映射失败或不可信，启用 LLM 推理")
        stage_2023_inferred = await llm_infer_figo_2023(session, case_id, raw_text_full, X, semaphore)
        if stage_2023_inferred != "unknown":
            # 剥离可能已存在的分子标注，提取基础分期
            base_inferred = stage_2023_inferred.split('m')[0] if 'm' in stage_2023_inferred else stage_2023_inferred
            stage_2023 = base_inferred
            if X["molecular_subtype"] != "unknown":
                _, stage_2023_full = apply_molecular_modification(base_inferred, X["molecular_subtype"])
            else:
                stage_2023_full = stage_2023_inferred

    X["stage_2023"] = stage_2023
    X["stage_2023_full"] = stage_2023_full

    # 生成 stage_2023 后立即调用决策树，将风险组和治疗建议存入 X
    X["esgo_risk_group"] = classify_esgo_risk(X)
    X["esgo_recommendation"] = recommend_adjuvant_therapy(
        X["esgo_risk_group"], X["molecular_subtype"], X["stage_2023"]
    )


    result_item = {
        "id": case_id,
        "X": X,
        "Y_structured": Y_structured,
        "Y_detail": Y_detail,
        "Y_text": Y_text,
        "llm_raw_output": llm_output,
        "truncated": truncated
    }

    pbar.update(1)
    return {"success": True, "result": result_item}


# ======================== 主异步流程 ========================
async def main_async():
    start_time = time.time()  # 记录开始时间

    if not INPUT_FILE.exists():
        print(f"错误：输入文件不存在 {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # 仅处理状态为“成功”的记录
    cases_to_process = [c for c in all_cases if c.get("status") == "成功"]
    total = len(cases_to_process)
    print(f"待处理病例数: {total}")

    results = []
    failed_cases = []

    # 清空原始输出文件
    if RAW_OUTPUT_FILE.exists():
        RAW_OUTPUT_FILE.unlink()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm_asyncio(total=total, desc="处理病历") as pbar:
            tasks = [process_one_case(session, case, semaphore, pbar) for case in cases_to_process]
            for coro in asyncio.as_completed(tasks):
                outcome = await coro
                if outcome["success"]:
                    results.append(outcome["result"])
                else:
                    failed_cases.append(outcome["failed"])

                # 定期保存
                if len(results) % SAVE_INTERVAL == 0:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"\n已保存 {len(results)} 条结果至 {OUTPUT_FILE}")

    # 最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"成功提取: {len(results)}")
    print(f"失败: {len(failed_cases)}")
    print(f"总 Token 消耗: {total_tokens}")
    print(f"失败样本已保存至: {FAILED_FILE}")
    print(f"原始LLM输出保存至: {RAW_OUTPUT_FILE}")
    print("=" * 50)

    # 打印详细统计报告
    if results:
        print_detailed_summary(results, failed_cases, total_tokens, elapsed_time)


    if len(failed_cases) / total > 0.1:
        logging.warning(f"失败率 {len(failed_cases) / total:.1%} 超过10%，请检查数据质量")

    print("全部处理完成！")


def print_missing_stats(results):
    from collections import defaultdict
    missing_count = defaultdict(int)
    total = len(results)
    for item in results:
        X = item.get("X", {})
        for field, value in X.items():
            if value in (None, "", "unknown", "none"):
                missing_count[field] += 1

    print("\n" + "=" * 60)
    print("字段缺失率统计（值 = unknown / None / 空）")
    print("=" * 60)
    for field in sorted(missing_count.keys()):
        rate = missing_count[field] / total * 100
        print(f"{field:25} : {missing_count[field]:3}/{total} ({rate:.1f}%)")
    print("=" * 60)

def print_detailed_summary(results, failed_cases, total_tokens, elapsed_time):
    """
    打印详细统计报告：成功率、字段分布、分期/分子分型分布、治疗决策、失败原因等
    """
    total_processed = len(results) + len(failed_cases)
    if total_processed == 0:
        print("\n没有处理任何数据。")
        return

    print("\n" + "=" * 70)
    print("📊 详细统计报告")
    print("=" * 70)

    # 1. 基本处理指标
    print(f"\n【处理概览】")
    print(f"  总处理病例数     : {total_processed}")
    print(f"  成功解析         : {len(results)} ({len(results)/total_processed*100:.1f}%)")
    print(f"  解析失败         : {len(failed_cases)} ({len(failed_cases)/total_processed*100:.1f}%)")
    print(f"  总 Token 消耗     : {total_tokens:,}")
    if len(results) > 0:
        print(f"  平均 Token/成功案例: {total_tokens/len(results):.0f}")
    print(f"  处理总耗时       : {elapsed_time:.2f} 秒")
    if len(results) > 0:
        print(f"  平均每例耗时     : {elapsed_time/len(results):.2f} 秒")

    # 2. 失败原因分类（如果有失败案例）
    if failed_cases:
        print(f"\n【失败原因分布】")
        reason_counts = {}
        for fc in failed_cases:
            reason = fc.get("reason", "未知原因")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            print(f"  {reason:20} : {count:3} ({count/len(failed_cases)*100:.1f}%)")

    # 3. 分期版本分布
    if results:
        print(f"\n【FIGO 版本分布】")
        version_counts = {"2009": 0, "2023": 0, "unknown": 0}
        for item in results:
            ver = item["X"].get("figo_version", "unknown")
            version_counts[ver] = version_counts.get(ver, 0) + 1
        for ver, cnt in version_counts.items():
            print(f"  FIGO {ver:6} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

        # 4. FIGO 2023 分期分布（转换后）
        print(f"\n【FIGO 2023 分期分布（转换后）】")
        stage_counts = {}
        for item in results:
            stage = item["X"].get("stage_2023_full", item["X"].get("stage_2023", "unknown"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        for stage in sorted(stage_counts.keys()):
            cnt = stage_counts[stage]
            print(f"  {stage:12} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

        # 5. 分子分型分布
        print(f"\n【分子分型分布】")
        mol_counts = {}
        for item in results:
            mol = item["X"].get("molecular_subtype", "unknown")
            mol_counts[mol] = mol_counts.get(mol, 0) + 1
        for mol in ["POLEmut", "MMRd", "NSMP", "p53abn", "unknown"]:
            cnt = mol_counts.get(mol, 0)
            if cnt > 0 or mol == "unknown":
                print(f"  {mol:10} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

        # 6. ESGO 风险组分布
        print(f"\n【ESGO 风险组分布】")
        risk_counts = {}
        for item in results:
            risk = item["X"].get("esgo_risk_group", "unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        for risk in ["低危", "中危", "高危", "晚期/转移", "unknown"]:
            cnt = risk_counts.get(risk, 0)
            if cnt > 0 or risk == "unknown":
                print(f"  {risk:12} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

        # 7. 治疗决策分布（来自 Y_structured）
        print(f"\n【治疗决策分布（推荐比例）】")
        therapies = ["radiotherapy", "chemotherapy", "targeted_therapy",
                     "immunotherapy", "hormone_therapy", "surgery"]
        for therapy in therapies:
            count_1 = sum(1 for item in results if item["Y_structured"].get(therapy) == 1)
            print(f"  {therapy:18} : {count_1:3} ({count_1/len(results)*100:.1f}%)")

        # 8. 治疗时机与方案摘要分布
        print(f"\n【治疗时机分布】")
        timing_counts = {}
        for item in results:
            timing = item["Y_detail"].get("timing", "unknown")
            timing_counts[timing] = timing_counts.get(timing, 0) + 1
        for t in ["neoadjuvant", "adjuvant", "palliative", "unknown"]:
            cnt = timing_counts.get(t, 0)
            if cnt > 0 or t == "unknown":
                print(f"  {t:12} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

        print(f"\n【治疗方案摘要分布】")
        summary_counts = {}
        for item in results:
            summ = item["Y_detail"].get("summary", "unknown")
            summary_counts[summ] = summary_counts.get(summ, 0) + 1
        for s in ["chemotherapy", "radiotherapy", "chemoradiation",
                  "surgery_only", "targeted", "immunotherapy", "hormone", "none", "unknown"]:
            cnt = summary_counts.get(s, 0)
            if cnt > 0 or s == "unknown":
                print(f"  {s:15} : {cnt:3} ({cnt/len(results)*100:.1f}%)")

    # 9. 字段缺失率统计（沿用原有函数，但稍作整理）
    print_missing_stats(results)

    print("\n" + "=" * 70)
    print("统计报告结束")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main_async())