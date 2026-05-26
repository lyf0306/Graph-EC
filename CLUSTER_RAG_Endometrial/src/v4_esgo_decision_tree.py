"""
ESGO 2025 风险分层与辅助治疗推荐
基于 FIGO 2023 分期 + 分子分型
输入：X 字典（包含 stage_2023, molecular_subtype, histology_type, grade, lvsi, lvsi_substantial, myometrial_invasion_ratio 等）
输出：风险组 + 治疗建议
"""

"""
ESGO 2025 风险分层 —— 兼容分子分型缺失的情况
"""

def classify_esgo_risk(X):
    """
    返回风险组：Low, Intermediate, High-Intermediate, High, Uncertain
    """
    stage = X.get("stage_2023", X.get("stage_raw", "unknown"))
    mol = X.get("molecular_subtype", "unknown")
    hist = X.get("histology_type", "unknown")
    grade = X.get("grade", "unknown")
    lvsi = X.get("lvsi", "unknown")
    lvsi_sub = X.get("lvsi_substantial", False)
    myo = X.get("myometrial_invasion_ratio", "unknown")

    # 高危组织学类型直接归入高危
    aggressive_hist = {"serous", "clear_cell", "carcinosarcoma", "undifferentiated", "mixed"}
    if hist in aggressive_hist:
        return "High"

    # ========== 分子分型已知 ==========
    if mol != "unknown":
        # POLEmut 早期均为低危
        if mol == "POLEmut":
            if stage in ("IA", "IB", "II", "I", "IIA", "IIB", "IIC",
                         "IA1", "IA2", "IA3"):
                return "Low"
            elif stage.startswith("III") or stage.startswith("IV"):
                return "Uncertain"

        # p53abn 高危驱动
        if mol == "p53abn":
            if stage in ("IA", "IB", "IIC", "IA1", "IA2", "IA3"):
                if myo == ">=50%" or lvsi_sub:
                    return "High"
                else:
                    return "High-Intermediate"
            elif stage.startswith("III") or stage.startswith("IV"):
                return "High"

        # MMRd 和 NSMP
        if stage in ("IA", "IA1", "IA2", "IA3"):
            if grade in ("G1", "G2") and not lvsi_sub:
                return "Low"
            else:
                return "Intermediate"
        elif stage in ("IB", "IC"):
            if grade in ("G1", "G2") and not lvsi_sub:
                return "Intermediate"
            else:
                return "High-Intermediate"
        elif stage in ("II", "IIA", "IIB"):
            if lvsi_sub:
                return "High-Intermediate"
            else:
                return "Intermediate"
        elif stage == "IIC":
            return "High"
        elif stage.startswith("III") or stage.startswith("IV"):
            return "High"

    # ========== 分子分型未知：回退到传统分层 ==========
    else:
        # 传统 FIGO 2009 风格风险分层
        if stage in ("IA", "IA1", "IA2", "IA3"):
            if grade in ("G1", "G2") and lvsi == "negative":
                return "Low"
            elif grade == "G3" or lvsi == "positive":
                return "Intermediate"
        elif stage in ("IB", "IC"):
            if grade in ("G1", "G2") and lvsi == "negative":
                return "Intermediate"
            else:
                return "High-Intermediate"
        elif stage in ("II", "IIA", "IIB"):
            return "High-Intermediate"
        elif stage.startswith("III") or stage.startswith("IV"):
            return "High"

    return "Uncertain"


def recommend_adjuvant_therapy(risk_group, molecular_subtype, stage_2023, surgery_done=True):
    """
    根据风险组返回辅助治疗建议
    """
    if not surgery_done:
        return "Primary surgery recommended before adjuvant therapy decision."

    if risk_group == "Low":
        return "No adjuvant therapy recommended."
    elif risk_group == "Intermediate":
        return ("Vaginal brachytherapy should be considered. "
                "No adjuvant therapy is an option for patients <60 years or low-grade tumors.")
    elif risk_group == "High-Intermediate":
        return ("External beam radiotherapy (EBRT) is recommended for optimal pelvic control. "
                "Vaginal brachytherapy is an alternative if lymph node staging was done and pN0.")
    elif risk_group == "High":
        if molecular_subtype == "MMRd" and stage_2023.startswith("III"):
            return ("EBRT with concurrent/sequential chemotherapy. "
                    "Consider adding immune checkpoint inhibitor for MMRd tumors.")
        else:
            return ("EBRT with concurrent and adjuvant chemotherapy, "
                    "or sequential chemotherapy and radiotherapy.")
    else:
        return ("Insufficient data for firm recommendation. Multidisciplinary discussion advised.")