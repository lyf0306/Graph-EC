
"""
合并症独立提取流水线
复用 v4_llm_pipeline 的异步框架
"""

import json
import asyncio
import aiohttp
from pathlib import Path
import re
import logging
import os
from tqdm.asyncio import tqdm_asyncio

# ===== 配置 =====
API_KEY = os.environ.get("LLM_API_KEY", "sk-your-api-key-here")
API_URL = "https://api.deepseek.com/v1/chat/completions"
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
INPUT_FILE = BASE_DIR / "data" / "extracted_output.json"
OUTPUT_FILE = BASE_DIR / "data" / "comorbidity_output_v4.json"
FAILED_FILE = BASE_DIR / "data" / "comorbidity_failed_v4.json"  # 新增：保存失败案例
MAX_CONCURRENT = 20
MAX_RETRIES = 3

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(BASE_DIR / "logs" / "comorbidity_extraction.log"),
              logging.StreamHandler()]
)

# ===== 合并症 Schema =====
COMORBIDITY_SCHEMA = {
    "glycemic_status": {"type": "ordinal", "values": {0: "正常", 1: "糖耐量异常/胰岛素抵抗/PCOS", 2: "糖尿病"},
                        "default": 0},
    "hypertension": {"type": "binary", "default": 0},
    "bmi_status": {"type": "ordinal", "values": {0: "正常", 1: "超重", 2: "肥胖"}, "default": 0},
    "hyperlipidemia": {"type": "binary", "default": 0},
    "anemia": {"type": "ordinal", "values": {0: "无", 1: "轻度", 2: "中重度"}, "default": 0},
    "hepatic_viral": {"type": "binary", "default": 0},
    "hepatic_dysfunction": {"type": "binary", "default": 0},
    "major_cv_risk": {"type": "binary", "default": 0},
    "hpv_status": {"type": "binary", "default": 0},
}

NOISE_KEYWORDS = ["肺结节", "甲状腺结节", "轻度胃炎", "子宫肌瘤", "子宫腺肌病", "宫颈囊肿"]


def build_comorbidity_prompt(case_id, raw_text):
    schema_desc = ""
    for key, cfg in COMORBIDITY_SCHEMA.items():
        if cfg["type"] == "ordinal":
            schema_desc += f'  "{key}": 0/1/2,\n'
        else:
            schema_desc += f'  "{key}": 0/1,\n'

    return f"""你是一名内科医生助手。请严格根据病历文本，提取患者的合并症与既往史信息。

忽略以下与肿瘤治疗禁忌无关的噪音词：{', '.join(NOISE_KEYWORDS)}。

字段说明：
- glycemic_status: 0=正常, 1=糖耐量异常/胰岛素抵抗/PCOS, 2=糖尿病
- hypertension: 0=无, 1=有
- bmi_status: 0=正常, 1=超重(BMI 24-28), 2=肥胖(BMI≥28)。若文中无BMI，根据描述判断
- hyperlipidemia: 0=无, 1=有(高脂血症/脂肪肝)
- anemia: 0=无, 1=轻度, 2=中重度
- hepatic_viral: 0=无, 1=有(乙肝/戊肝)  ← 极高化疗警戒
- hepatic_dysfunction: 0=无, 1=有(肝功能异常)
- major_cv_risk: 0=无, 1=有(脑梗/冠心病/肾功能不全)  ← 最高权重
- hpv_status: 0=无, 1=有(HPV感染/HPV阳性)  ← 影响宫颈癌筛查与免疫治疗决策

输出格式（只输出JSON，不要任何额外文字）：
{{
  "id": "{case_id}",
{schema_desc.strip()}
}}

文本：
{raw_text[:6500] if raw_text else ""}
"""


def extract_json_robust(raw_output):
    """改进的JSON提取，处理嵌套和多个JSON块"""
    if not raw_output:
        return None

    # 尝试直接解析
    raw_output = raw_output.strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # 尝试提取最外层JSON（非贪婪匹配最外层大括号）
    # 使用递归下降思路：找到第一个 {，然后匹配对应的 }
    stack = []
    start = -1
    for i, char in enumerate(raw_output):
        if char == '{':
            if not stack:
                start = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start != -1:
                    try:
                        candidate = raw_output[start:i + 1]
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

    # 备选：查找第一个 [ ... ] 如果LLM错误地返回了数组
    if raw_output.startswith('['):
        try:
            arr = json.loads(raw_output)
            if isinstance(arr, list) and len(arr) > 0:
                return arr[0]
        except:
            pass

    return None


def normalize_comorbidity(parsed):
    """标准化合并症字段，缺失时用默认值"""
    if not isinstance(parsed, dict):
        return {key: cfg["default"] for key, cfg in COMORBIDITY_SCHEMA.items()}

    result = {}
    for key, cfg in COMORBIDITY_SCHEMA.items():
        val = parsed.get(key, cfg["default"])
        if cfg["type"] == "ordinal":
            try:
                result[key] = int(val) if int(val) in (0, 1, 2) else cfg["default"]
            except (ValueError, TypeError):
                result[key] = cfg["default"]
        else:
            result[key] = 1 if val in (1, "1", True, "true", "yes", "是", "有") else 0
    return result


async def call_llm_async(session, prompt, case_id, semaphore):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500  # 限制输出长度，JSON很短
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with session.post(API_URL, headers=headers, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    content = result["choices"][0]["message"]["content"]
                    logging.debug(f"{case_id} 原始输出: {content[:200]}...")
                    return content
            except Exception as e:
                logging.warning(f"{case_id} LLM调用失败（第{attempt + 1}次）: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        return None


async def process_one_case(session, case, semaphore, pbar):
    # 健壮性：检查必需字段
    case_id = case.get("id", "unknown")
    raw_text = case.get("raw_text", "")

    if not raw_text:
        logging.warning(f"{case_id} 缺少 raw_text 字段")
        pbar.update(1)
        return {"success": False, "id": case_id, "error": "missing raw_text"}

    prompt = build_comorbidity_prompt(case_id, raw_text)
    llm_output = await call_llm_async(session, prompt, case_id, semaphore)

    if llm_output is None:
        pbar.update(1)
        return {"success": False, "id": case_id, "raw_output": None, "error": "llm_call_failed"}

    parsed = extract_json_robust(llm_output)
    if not parsed:
        pbar.update(1)
        return {"success": False, "id": case_id, "raw_output": llm_output, "error": "json_parse_failed"}

    como_features = normalize_comorbidity(parsed)
    pbar.update(1)
    return {"success": True, "id": case_id, "X": como_features}


async def main_async():
    # 确保目录存在
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(exist_ok=True)

    # 检查输入文件
    if not INPUT_FILE.exists():
        logging.error(f"输入文件不存在: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_cases = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"输入文件JSON格式错误: {e}")
        return

    # 过滤有效病例，并检查字段存在性
    cases = [c for c in all_cases if c.get("status") == "成功" and "raw_text" in c]
    logging.info(f"总病例数: {len(all_cases)}, 待处理病例数: {len(cases)}")

    if not cases:
        logging.warning("没有符合条件的病例需要处理")
        return

    results = []
    failed_cases = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm_asyncio(total=len(cases), desc="提取合并症") as pbar:
            tasks = [process_one_case(session, case, semaphore, pbar) for case in cases]

            for coro in asyncio.as_completed(tasks):
                outcome = await coro
                if outcome["success"]:
                    results.append({"id": outcome["id"], "X": outcome["X"]})
                else:
                    failed_cases.append({
                        "id": outcome["id"],
                        "error": outcome.get("error"),
                        "raw_output": outcome.get("raw_output")
                    })

    # 保存成功结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存失败案例以便后续处理
    if failed_cases:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_cases, f, ensure_ascii=False, indent=2)
        logging.warning(f"处理完成：成功 {len(results)}，失败 {len(failed_cases)}。失败案例保存至 {FAILED_FILE}")
    else:
        logging.info(f"全部成功处理：{len(results)} 例")

    # 简单统计
    if results:
        logging.info("各合并症阳性率统计：")
        for key in COMORBIDITY_SCHEMA.keys():
            count = sum(1 for r in results if r["X"].get(key, 0) > 0)
            logging.info(f"  {key}: {count}/{len(results)} ({count / len(results) * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main_async())