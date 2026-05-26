# 合并症 Schema
COMORBIDITY_SCHEMA = {
    "glycemic_status": {"type": "ordinal", "values": {0:"正常",1:"糖耐量异常/胰岛素抵抗/PCOS",2:"糖尿病"}, "default":0},
    "hypertension": {"type": "binary", "default":0},
    "bmi_status": {"type": "ordinal", "values": {0:"正常",1:"超重",2:"肥胖"}, "default":0},
    "hyperlipidemia": {"type": "binary", "default":0},
    "anemia": {"type": "ordinal", "values": {0:"无",1:"轻度",2:"中重度"}, "default":0},
    "hepatic_viral": {"type": "binary", "default":0},
    "hepatic_dysfunction": {"type": "binary", "default":0},
    "major_cv_risk": {"type": "binary", "default":0}
}

# 噪音关键词列表，这些词可能在病历中出现，但与肿瘤治疗禁忌无关，提取时应忽略它们以减少误判
NOISE_KEYWORDS = ["肺结节", "甲状腺结节", "轻度胃炎", "子宫肌瘤", "子宫腺肌病", "HPV感染", "宫颈囊肿"]