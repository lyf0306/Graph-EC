"""
脚本说明：
从肿瘤委员会讨论记录中抽取结构化临床信息，调用 DeepSeek API 进行信息提取，
并输出标准化的 JSON 数据，同时记录原始 LLM 输出与失败样本。

主要流程：
1. 读取 extracted_output.json 中的原始病历文本。
2. 对每条文本进行智能截断（保留关键词行），构造 Prompt 调用 LLM。
3. 解析 LLM 返回的 JSON，进行严格的字段值校验与归一化。
4. 保存结构化结果、原始 LLM 输出及失败样本。
5. 输出处理统计与字段缺失率。
"""

import json
import time
import re
import requests
from tqdm import tqdm
from pathlib import Path
import logging
import openai  # 新增
from datetime import datetime

# ======================== 配置参数 ========================
# API_KEY = "sk-a92d8d938ced4093a0176cc6cb0e7d60"  # DeepSeek API 密钥（实际使用时建议从环境变量读取）
# API_URL = "https://api.deepseek.com/v1/chat/completions"
VLLM_API_KEY = "EMPTY"
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL_NAME = "OriClinical"

client = openai.OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)



# 文件路径定义（基于当前脚本所在目录的上一级 data/ 和 logs/ 目录）
INPUT_FILE = Path(__file__).parent.parent / "data" / "extracted_output.json"  # 输入：包含原始文本的JSON
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "structured_output.json"  # 输出：最终结构化结果
FAILED_FILE = Path(__file__).parent.parent / "data" / "failed_parsing.json"  # 输出：解析失败样本
RAW_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "llm_raw_outputs.jsonl"  # 输出：原始LLM输出（便于调试）
LOG_FILE = Path(__file__).parent.parent / "logs" / "structuring.log"    # 日志文件

SLEEP_TIME = 1  # 请求间隔（秒），避免API限流
SAVE_INTERVAL = 10  # 每处理多少条数据保存一次中间结果
MAX_RETRIES = 3  # API调用最大重试次数
MAX_INPUT_LEN = 6000  # 输入文本最大长度（字符数），超过则截断

# 确保日志目录存在
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ======================== 日志配置 ========================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

"""
分析：日志仅写入文件，未输出到控制台（避免干扰进度条）。可根据需要添加 StreamHandler。
"""


# ======================== 智能截断函数 ========================
def smart_truncate(text, max_len=6000):
    """
    对输入文本进行智能截断，优先保留包含医学关键词的行。
    当文本超过 max_len 时，先提取所有包含关键词的行，再尽可能补充其他行，最后按长度截断。
    """
    if len(text) <= max_len:
        return text

    # 定义关键医学词汇（用于识别重要信息行）
    keywords = [
        "诊断", "病理", "分期", "免疫组化", "分子分型",
        "手术", "术后", "淋巴结", "肌层", "LVSI", "p53", "MMR",
        "治疗", "化疗", "放疗", "靶向", "免疫", "激素",
        "建议", "TB", "肿瘤委员会", "讨论", "结论", "意见"
    ]

    important_lines = []  # 包含关键词的行
    other_lines = []  # 其他行

    for line in text.split('\n'):
        if any(k in line for k in keywords):
            important_lines.append(line)
        else:
            other_lines.append(line)

    # 先保留所有重要行
    selected = important_lines[:]
    # 如果还有空间，继续添加其他行
    if len('\n'.join(selected)) < max_len:
        for line in other_lines:
            selected.append(line)
            if len('\n'.join(selected)) >= max_len:
                break

    result = '\n'.join(selected)
    if len(result) > max_len:
        result = result[:max_len]  # 极少数情况仍超长，直接截断

    if len(result) < len(text):
        result += "\n...[文本过长已智能截断]"  # 添加截断标记，提醒LLM输入不完整

    return result


"""
分析：
该函数通过关键词优先保留可能含有重要诊断、治疗信息的行，尽量避免因截断丢失关键内容。
对于TB讨论记录，这样的策略是合理的，因为核心决策信息往往集中在包含上述关键词的段落中。
截断后添加提示语，有助于LLM意识到信息不完整，避免强行编造。
"""

# ======================== 允许值集合（枚举字段校验用） ========================
ALLOWED_VALUES = {
    "menopause": {"yes", "no", "unknown"},
    "histology_type": {"endometrioid", "serous", "clear_cell", "carcinosarcoma", "mixed", "unknown"},
    "grade": {"G1", "G2", "G3", "unknown"},
    "stage": {"IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB", "unknown"},
    "myometrial_invasion_ratio": {"<50%", ">=50%", "unknown"},
    "cervical_involvement": {"none", "glandular", "stromal", "unknown"},
    "lvsi": {"positive", "negative", "unknown"},
    "peritoneal_cytology": {"negative", "positive", "unknown"},
    "p53": {"wild", "mutant", "unknown"},
    "mmr": {"proficient", "deficient", "unknown"},
    "molecular_subtype": {"POLEmut", "MSI-H", "NSMP", "p53abn", "unknown"},
    "timing": {"neoadjuvant", "adjuvant", "palliative", "unknown"},
    "summary": {"chemotherapy", "radiotherapy", "chemoradiation", "surgery_only", "targeted", "immunotherapy",
                "hormone", "none", "unknown"}
}

# 全局 token 消耗统计（供最后汇总）
total_tokens = 0


# ======================== 字段校验与归一化函数 ========================
def validate_enum(value, allowed_set, default="unknown"):
    """
    校验枚举值是否在允许集合中，若不是则返回默认值并记录警告。
    支持大小写不敏感匹配及特殊处理 "none"。
    """
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
    """
    将输入转换为 0 或 1。
    支持数字、布尔值，以及 "yes"/"no"/"有"/"无" 等常见表达。
    """
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
    """将输入转为整数，失败则返回默认值。"""
    if value is None:
        return default
    try:
        return int(float(value))
    except:
        logging.warning(f"无效数字: {value}，将设为 {default}")
        return default


def validate_lymph_node(value):
    """
    校验淋巴结字段格式：支持 "positive"/"negative"/"unknown" 或形如 "1/10" 的计数。
    """
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
    """
    提取肌层浸润深度的数值（mm），返回浮点数或 None。
    若字符串中包含数字则提取第一个数字。
    """
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
    """检查解析后的 JSON 是否包含顶层必要字段。"""
    required_top = ["X", "Y_structured", "Y_detail", "Y_text"]
    for field in required_top:
        if field not in parsed:
            return False
    return True


"""
分析：
校验函数对 LLM 输出进行了二次保障，确保最终数据格式统一且可计算。
即使 LLM 未严格遵守枚举约束，也能通过归一化得到合理的默认值，提高后续处理的鲁棒性。
"""


# ======================== JSON 提取函数（健壮版） ========================
def extract_json_robust(raw_output):
    """
    从 LLM 返回的文本中提取 JSON 对象。
    首先尝试直接解析，失败则通过正则匹配最外层花括号并检查括号平衡。
    """
    if raw_output is None:
        return None
    try:
        return json.loads(raw_output)
    except:
        pass

    # 正则匹配第一个完整 JSON 对象（贪婪匹配最外层花括号）
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        candidate = match.group()
        # 简单括号平衡检查，避免截断不完整 JSON
        if candidate.count('{') == candidate.count('}'):
            try:
                return json.loads(candidate)
            except:
                pass
    logging.warning(f"JSON解析失败，原始输出前200字符: {raw_output[:200]}")
    return None


"""
分析：
LLM 有时会在 JSON 前后附加解释文字，直接 json.loads 会失败。
此函数通过正则匹配大括号内容并验证平衡，提高了容错能力，适合生产环境。
"""


# ======================== LLM 调用函数 ========================
def call_llm(prompt):
    """调用本地 vLLM 模型，带重试机制"""
    global total_tokens
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=6000,
            )
            # 累计 token 消耗
            total_tokens += response.usage.total_tokens
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"LLM调用失败（第{attempt + 1}次）: {e}")
            time.sleep(2)
    logging.error("LLM调用最终失败")
    return None


"""
分析：
调用失败时的日志记录有助于排查网络或 API 问题。累计 token 统计可估算本次处理成本。
"""


# ======================== Prompt 构造 ========================
def build_prompt(case_id, raw_text):
    return f"""
你是一名妇科肿瘤数据结构化专家。请仔细阅读肿瘤委员会讨论记录，提取以下信息，并严格按照要求输出 JSON，不要添加任何额外文字。

### 需要提取的信息域（以下为字段含义说明，最终输出必须为 JSON）

**基本体征**
- 年龄：[数字]
- 绝经状态：[yes / no / unknown]

**所有合并症与既往史**
- 高血压：[0 或 1]
- 糖尿病：[0 或 1]
- 肥胖 (BMI≥30)：[0 或 1]

**既定分期**
- FIGO 分期：[IA / IB / II / IIIA / IIIB / IIIC1 / IIIC2 / IVA / IVB / unknown]

**病理与转移特征**
- 组织学类型：[endometrioid / serous / clear_cell / carcinosarcoma / mixed / unknown]
- 组织学分级：[G1 / G2 / G3 / unknown]
  *注意：低分化/高级别 → G3；中分化 → G2；高分化/低级别 → G1*
- 肌层浸润比例：[<50% / >=50% / unknown]
- 肌层浸润深度(mm)：[数字或 null]
- 宫颈受累情况：[none / glandular / stromal / unknown]
  *注意：若未提及宫颈受累或侵犯，默认为 none*
- 脉管癌栓(LVSI)：[positive / negative / unknown]
  *注意：若报告提及“脉管浸润”、“血管侵犯”、“LVSI+”等均为 positive；若明确阴性或未提及可设为 negative*
- 附件受累：[0 或 1]
- 盆腔淋巴结状态：[negative / positive 或计数如 0/10 / unknown]
- 腹主动脉旁淋巴结状态：[格式同上]
- 腹腔细胞学：[negative / positive / unknown]
- p53 状态：[wild / mutant / unknown]
  *注意：过表达/突变型 → mutant；野生型/无异常 → wild*
- MMR 状态：[proficient / deficient / unknown]
  *注意：若 MLH1、PMS2、MSH2、MSH6 均为完整/阳性/表达 → proficient；任一缺失/阴性 → deficient*
- 分子分型：[POLEmut / MSI-H / NSMP / p53abn / unknown]
  *注意：若文本中明确提到"结果未出"、"待回报"、"未出结果"、"尚未回报"等，或根本未提及分子分型结果，请设为 unknown。切勿根据其他信息推测分子分型值。*
- 组织学详细描述（自由文本）：[可补充其他病理细节]

**治疗决策**
- 放疗：[0 或 1]
- 化疗：[0 或 1]
- 靶向治疗：[0 或 1]
- 免疫治疗：[0 或 1]
- 激素治疗：[0 或 1]
- 手术：[0 或 1]

**治疗方案详情**
- 治疗时机：[neoadjuvant / adjuvant / palliative / unknown]
- 具体方案（自由文本）：[如 TC 方案化疗 6 周期]
- 治疗方案摘要：[chemotherapy / radiotherapy / chemoradiation / surgery_only / targeted / immunotherapy / hormone / none / unknown]

**治疗建议原文缩略**
- 原文描述（不超过100字）：[摘录核心建议语句]

---

### 输出格式（必须严格遵循以下 JSON 结构）

{{
  "id": "{case_id}",
  "X": {{
    "age": 0,
    "menopause": "",
    "histology_type": "",
    "histology_detail": "",
    "grade": "",
    "stage": "",
    "myometrial_invasion_ratio": "",
    "myometrial_invasion_depth": null,
    "cervical_involvement": "",
    "lvsi": "",
    "lymph_node_pelvic": "",
    "lymph_node_paraaortic": "",
    "peritoneal_cytology": "",
    "adnexal_involvement": 0,
    "p53": "",
    "mmr": "",
    "molecular_subtype": "",
    "hypertension": 0,
    "diabetes": 0,
    "obesity": 0
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

---

### 示例1（辅助放化疗）

病历片段：
"...51岁，已绝经。术后病理：子宫内膜样癌，G3，FIGO IIIC2期，深肌层浸润，LVSI阳性，盆腔淋巴结1/10转移，腹主动脉旁淋巴结1/13转移。p53野生型，MMR完整。肿瘤委员会建议术后辅助TC方案化疗6周期，联合外照射放疗。"

输出 JSON：
{{
  "id": "{case_id}",
  "X": {{
    "age": 51,
    "menopause": "yes",
    "histology_type": "endometrioid",
    "histology_detail": "",
    "grade": "G3",
    "stage": "IIIC2",
    "myometrial_invasion_ratio": ">=50%",
    "myometrial_invasion_depth": null,
    "cervical_involvement": "unknown",
    "lvsi": "positive",
    "lymph_node_pelvic": "1/10",
    "lymph_node_paraaortic": "1/13",
    "peritoneal_cytology": "unknown",
    "adnexal_involvement": 0,
    "p53": "wild",
    "mmr": "proficient",
    "molecular_subtype": "unknown",
    "hypertension": 0,
    "diabetes": 0,
    "obesity": 0
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
    "regimen": "TCx6 + EBRT",
    "summary": "chemoradiation"
  }},
  "Y_text": "TC方案化疗6周期联合外照射放疗"
}}

### 示例2（仅手术，不辅助治疗）

病历片段：
"...56岁，已绝经。术后病理：浆液性癌，FIGO IA期，局限于宫腔，无肌层浸润，LVSI阴性，盆腔及腹主动脉旁淋巴结均为阴性。TB决定随访观察，无需辅助治疗。"

输出 JSON：
{{
  "id": "{case_id}",
  "X": {{
    "age": 56,
    "menopause": "yes",
    "histology_type": "serous",
    "histology_detail": "",
    "grade": "unknown",
    "stage": "IA",
    "myometrial_invasion_ratio": "unknown",
    "myometrial_invasion_depth": null,
    "cervical_involvement": "unknown",
    "lvsi": "negative",
    "lymph_node_pelvic": "negative",
    "lymph_node_paraaortic": "negative",
    "peritoneal_cytology": "unknown",
    "adnexal_involvement": 0,
    "p53": "unknown",
    "mmr": "unknown",
    "molecular_subtype": "unknown",
    "hypertension": 0,
    "diabetes": 0,
    "obesity": 0
  }},
  "Y_structured": {{
    "radiotherapy": 0,
    "chemotherapy": 0,
    "targeted_therapy": 0,
    "immunotherapy": 0,
    "hormone_therapy": 0,
    "surgery": 1
  }},
  "Y_detail": {{
    "timing": "adjuvant",
    "regimen": "",
    "summary": "none"
  }},
  "Y_text": "no treatment"
}}

### 示例3（辅助化疗，无放疗）

病历片段：
"...63岁，未绝经。术后病理：子宫内膜样癌，G2，FIGO IB期，肌层浸润<50%，未见LVSI，淋巴结0/10。免疫组化：p53野生型，MMR完整。TB建议辅助卡铂+紫杉醇化疗3周期。"

输出 JSON：
{{
  "id": "{case_id}",
  "X": {{
    "age": 63,
    "menopause": "no",
    "histology_type": "endometrioid",
    "histology_detail": "",
    "grade": "G2",
    "stage": "IB",
    "myometrial_invasion_ratio": "<50%",
    "myometrial_invasion_depth": null,
    "cervical_involvement": "unknown",
    "lvsi": "negative",
    "lymph_node_pelvic": "0/10",
    "lymph_node_paraaortic": "unknown",
    "peritoneal_cytology": "unknown",
    "adnexal_involvement": 0,
    "p53": "wild",
    "mmr": "proficient",
    "molecular_subtype": "unknown",
    "hypertension": 0,
    "diabetes": 0,
    "obesity": 0
  }},
  "Y_structured": {{
    "radiotherapy": 0,
    "chemotherapy": 1,
    "targeted_therapy": 0,
    "immunotherapy": 0,
    "hormone_therapy": 0,
    "surgery": 1
  }},
  "Y_detail": {{
    "timing": "adjuvant",
    "regimen": "卡铂+紫杉醇 x3",
    "summary": "chemotherapy"
  }},
  "Y_text": "辅助卡铂+紫杉醇化疗3周期"
}}

---

现在请处理以下病历文本，只输出 JSON，不要有任何额外内容：

文本：
\"\"\"
{raw_text}
\"\"\"
"""

"""
分析：
Prompt 设计要点：
1. 明确角色与任务。
2. 强调输出纯 JSON，禁止额外文字。
3. 给出每个字段的可选值集合，并附示例。
4. 提供两个典型场景的示例，帮助 LLM 理解期望输出。
5. 输入文本用三引号包裹，避免转义问题。
"""


# ======================== 后处理标准化函数 ========================
def normalize_X(X_raw):
    """对 X 部分的每个字段进行类型校验与标准化。"""
    return {
        "age": validate_number(X_raw.get("age")),
        "menopause": validate_enum(X_raw.get("menopause"), ALLOWED_VALUES["menopause"]),
        "histology_type": validate_enum(X_raw.get("histology_type"), ALLOWED_VALUES["histology_type"]),
        "histology_detail": X_raw.get("histology_detail", ""),
        "grade": validate_enum(X_raw.get("grade"), ALLOWED_VALUES["grade"]),
        "stage": validate_enum(X_raw.get("stage"), ALLOWED_VALUES["stage"]),
        "myometrial_invasion_ratio": validate_enum(X_raw.get("myometrial_invasion_ratio"),
                                                   ALLOWED_VALUES["myometrial_invasion_ratio"]),
        "myometrial_invasion_depth": validate_myometrial_invasion_depth(X_raw.get("myometrial_invasion_depth")),
        "cervical_involvement": validate_enum(X_raw.get("cervical_involvement"),
                                              ALLOWED_VALUES["cervical_involvement"]),
        "lvsi": validate_enum(X_raw.get("lvsi"), ALLOWED_VALUES["lvsi"]),
        "lymph_node_pelvic": validate_lymph_node(X_raw.get("lymph_node_pelvic")),
        "lymph_node_paraaortic": validate_lymph_node(X_raw.get("lymph_node_paraaortic")),
        "peritoneal_cytology": validate_enum(X_raw.get("peritoneal_cytology"), ALLOWED_VALUES["peritoneal_cytology"]),
        "adnexal_involvement": validate_binary(X_raw.get("adnexal_involvement")),
        "p53": validate_enum(X_raw.get("p53"), ALLOWED_VALUES["p53"]),
        "mmr": validate_enum(X_raw.get("mmr"), ALLOWED_VALUES["mmr"]),
        "molecular_subtype": validate_enum(X_raw.get("molecular_subtype"), ALLOWED_VALUES["molecular_subtype"]),
        "hypertension": validate_binary(X_raw.get("hypertension")),
        "diabetes": validate_binary(X_raw.get("diabetes")),
        "obesity": validate_binary(X_raw.get("obesity"))
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
    """清理 Y_text 字段，去除常见冗余前缀并截断至100字符。"""
    if not text or text == "unknown":
        return "unknown"
    cleaned = text.strip()
    # 常见前缀（多为套话，可去除以突出核心内容）
    prefixes = ["根据术后病理", "结合文献回顾及NCCN指南", "肿瘤委员会讨论决定", "根据NCCN指南", "建议", "决定"]
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):].lstrip("，").lstrip(",").strip()
    if len(cleaned) > 100:
        cleaned = cleaned[:100] + "..."
    return cleaned


"""
分析：
标准化函数保证了最终输出数据的一致性，并为后续统计分析提供了可靠基础。
"""


# ======================== 字段缺失率统计函数 ========================
def print_missing_stats(results):
    """统计 X 中各字段值为 unknown / None / 空字符串的比例，并打印。"""
    from collections import defaultdict
    missing_count = defaultdict(int)
    total = len(results)
    if total == 0:
        print("无有效结果，无法统计缺失率")
        return

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


"""
分析：
缺失率统计有助于评估 LLM 提取质量，识别哪些字段难以从文本中获取，可指导后续 Prompt 优化或后处理规则补充。
"""


# ======================== 主流程 ========================
def main():
    global total_tokens
    if not INPUT_FILE.exists():
        print(f"错误：输入文件不存在 {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # 测试模式：只处理前50条（可根据需要注释掉）
    # all_cases = all_cases[:50]

    results = []
    failed_cases = []
    total = len(all_cases)

    # 清空原始输出文件（后续以追加模式写入）
    if RAW_OUTPUT_FILE.exists():
        RAW_OUTPUT_FILE.unlink()

    for idx, case in enumerate(tqdm(all_cases, desc="处理病历")):
        # 跳过状态不为成功的记录（step1 可能标记失败）
        if case.get("status") != "成功":
            logging.info(f"跳过 {case.get('id')}，状态不为成功")
            continue

        case_id = case["id"]
        raw_text_full = case["raw_text"]

        # ===== 智能截断 =====
        truncated = False
        if len(raw_text_full) > MAX_INPUT_LEN:
            raw_text = smart_truncate(raw_text_full, MAX_INPUT_LEN)
            truncated = True
            logging.warning(f"{case_id} 文本过长({len(raw_text_full)}字符)，已智能截断至{len(raw_text)}字符")
        else:
            raw_text = raw_text_full

        print(f"\n[{idx + 1}/{total}] 处理: {case_id} {'(截断)' if truncated else ''}")

        # 调用 LLM
        prompt = build_prompt(case_id, raw_text)
        llm_output = call_llm(prompt)

        # 保存原始输出（jsonl 格式，便于后续分析）
        with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as rf:
            rf.write(json.dumps({"id": case_id, "raw_output": llm_output}, ensure_ascii=False) + "\n")

        # 解析 JSON
        parsed = extract_json_robust(llm_output)

        # 检查必要字段
        if not parsed or not check_required_fields(parsed):
            reason = "JSON解析失败或字段缺失"
            logging.error(f"{case_id} {reason}")
            failed_cases.append({
                "id": case_id,
                "raw_text": raw_text_full,
                "llm_output": llm_output,
                "reason": reason
            })
            continue

        # 可选：核对返回的 ID 是否与输入一致（若不匹配仅警告，不视为失败）
        if parsed.get("id") != case_id:
            logging.warning(f"{case_id} ID不匹配: {parsed.get('id')}")

        # 提取各子部分
        X_raw = parsed.get("X", {})
        Y_structured_raw = parsed.get("Y_structured", {})
        Y_detail_raw = parsed.get("Y_detail", {})
        Y_text_raw = parsed.get("Y_text", "")

        # 标准化
        X = normalize_X(X_raw)
        Y_structured = normalize_Y_structured(Y_structured_raw)
        Y_detail = normalize_Y_detail(Y_detail_raw)
        Y_text = clean_y_text(Y_text_raw)

        # 组装最终结果，同时保留原始 LLM 输出供调试
        result_item = {
            "id": case_id,
            "X": X,
            "Y_structured": Y_structured,
            "Y_detail": Y_detail,
            "Y_text": Y_text,
            "llm_raw_output": llm_output
        }
        results.append(result_item)

        # 定期保存中间结果（防止中断丢失数据）
        if (idx + 1) % SAVE_INTERVAL == 0 or (idx + 1) == total:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n已保存 {len(results)} 条结果至 {OUTPUT_FILE}")

        time.sleep(SLEEP_TIME)  # 控制请求频率

    # 最终保存（覆盖之前的中间保存，确保一致）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存失败样本
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print("\n" + "=" * 50)
    print(f"总处理病例数: {total}")
    print(f"成功提取: {len(results)}")
    print(f"失败: {len(failed_cases)}")
    print(f"总 Token 消耗: {total_tokens}")
    print(f"失败样本已保存至: {FAILED_FILE}")
    print(f"原始LLM输出保存至: {RAW_OUTPUT_FILE}")
    print("=" * 50)

    # 字段缺失率统计（仅对成功提取的样本）
    if results:
        print_missing_stats(results)

    # 如果失败率过高，发出警告（便于及时检查 Prompt 或 API 状态）
    if len(failed_cases) / total > 0.1:
        logging.warning(f"失败率 {len(failed_cases) / total:.1%} 超过10%，请检查数据质量")

    print("全部处理完成！")


if __name__ == "__main__":
    main()