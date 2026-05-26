# utils/trans_format.py
"""
患者特征自然语言格式化工具（适配新 Schema V4）
"""
import pandas as pd

def format_patient_desc(row):
    """
    将包含患者特征的一行数据（Series 或 dict-like）转换为自然语言描述。
    适用于相似患者展示或当前患者描述。
    列名格式：X_xxx
    """
    # 年龄
    age_val = row.get('X_age', 0)
    if pd.notna(age_val):
        age = int(float(age_val)) if isinstance(age_val, (int, float, str)) else '未知'
    else:
        age = '未知'

    # 绝经状态
    meno_map = {'yes': '已绝经', 'no': '未绝经', 'unknown': '未知'}
    meno = meno_map.get(row.get('X_menopause'), '未知')

    # 组织学类型
    hist_map = {
        'endometrioid': '子宫内膜样癌',
        'serous': '浆液性癌',
        'clear_cell': '透明细胞癌',
        'carcinosarcoma': '癌肉瘤',
        'mixed': '混合型',
        'unknown': '未知类型'
    }
    hist = hist_map.get(row.get('X_histology_type'), row.get('X_histology_type', '未知'))

    # 分级
    grade_map = {'G1': 'G1', 'G2': 'G2', 'G3': 'G3', 'unknown': '未知'}
    grade = grade_map.get(row.get('X_grade'), row.get('X_grade', '未知'))

    # ========= 修复分期取值 =========
    stage_2023 = row.get('X_stage_2023')
    stage_raw = row.get('X_stage_raw')
    # 处理可能的 NaN
    if pd.isna(stage_2023):
        stage_2023 = None
    if pd.isna(stage_raw):
        stage_raw = None

    if stage_2023 and stage_2023 != 'unknown':
        stage = stage_2023
    elif stage_raw and stage_raw != 'unknown':
        stage = stage_raw
    else:
        stage = '未知'
    # =================================

    # 肌层浸润比例
    myo_map = {'<50%': '＜50%', '>=50%': '≥50%', 'unknown': '未知'}
    myo = myo_map.get(row.get('X_myometrial_invasion_ratio'), '未知')

    # LVSI
    lvsi_map = {'positive': '有', 'negative': '无', 'unknown': '未知'}
    lvsi = lvsi_map.get(row.get('X_lvsi'), '未知')
    lvsi_sub = row.get('X_lvsi_substantial', 0)
    if lvsi_sub == 1:
        lvsi += "（显著）"

    # 盆腔淋巴结
    lymph_pelvic = row.get('X_lymph_node_pelvic', '未知')
    if lymph_pelvic == 'negative':
        lymph_pelvic = '阴性'
    elif lymph_pelvic == 'positive':
        lymph_pelvic = '阳性'

    # 腹主动脉旁淋巴结
    lymph_para = row.get('X_lymph_node_paraaortic', '未知')
    if lymph_para == 'negative':
        lymph_para = '阴性'
    elif lymph_para == 'positive':
        lymph_para = '阳性'
    elif lymph_para == 'unknown':
        lymph_para = '未知'

    # 宫颈受累
    cervical_map = {'none': '无', 'glandular': '腺体受累', 'stromal': '间质受累', 'unknown': '未知'}
    cervical = cervical_map.get(row.get('X_cervical_involvement'), '未知')

    # p53
    p53_map = {'wild': '野生型', 'mutant': '突变型', 'unknown': '未知'}
    p53 = p53_map.get(row.get('X_p53'), '未知')

    # MMR
    mmr_map = {'proficient': '正常(pMMR)', 'deficient': '缺陷(dMMR)', 'unknown': '未知'}
    mmr = mmr_map.get(row.get('X_mmr'), '未知')

    # 分子分型
    mol_map = {
        'POLEmut': 'POLE突变型',
        'MMRd': '错配修复缺陷型',
        'NSMP': '无特殊分子谱型',
        'p53abn': 'p53异常型',
        'unknown': '未知'
    }
    mol = mol_map.get(row.get('X_molecular_subtype'), row.get('X_molecular_subtype', '未知'))

    # ESGO 风险组
    risk = row.get('X_esgo_risk_group', '未知')
    risk_display = risk if risk != 'unknown' else '未评估'

    # 合并症
    comorb_list = []
    gly = row.get('X_glycemic_status', 0)
    if gly == 2:
        comorb_list.append('糖尿病')
    elif gly == 1:
        comorb_list.append('糖耐量异常/胰岛素抵抗')
    if row.get('X_hypertension', 0) == 1:
        comorb_list.append('高血压')
    bmi = row.get('X_bmi_status', 0)
    if bmi == 2:
        comorb_list.append('肥胖')
    elif bmi == 1:
        comorb_list.append('超重')
    if row.get('X_hyperlipidemia', 0) == 1:
        comorb_list.append('高脂血症/脂肪肝')
    anemia = row.get('X_anemia', 0)
    if anemia == 2:
        comorb_list.append('中重度贫血')
    elif anemia == 1:
        comorb_list.append('轻度贫血')
    if row.get('X_hepatic_viral', 0) == 1:
        comorb_list.append('乙肝/戊肝')
    if row.get('X_hepatic_dysfunction', 0) == 1:
        comorb_list.append('肝功能异常')
    if row.get('X_major_cv_risk', 0) == 1:
        comorb_list.append('冠心病/脑梗/肾衰')
    if row.get('X_hpv_status', 0) == 1:
        comorb_list.append('HPV感染')

    comorb_str = '，'.join(comorb_list) if comorb_list else '无特殊合并症'

    adnexal = "是" if row.get('X_adnexal_involvement', 0) == 1 else "否"
    cyto_map = {'negative': '阴性', 'positive': '阳性', 'unknown': '未知'}
    cyto = cyto_map.get(row.get('X_peritoneal_cytology'), '未知')
    depth = row.get('X_myometrial_invasion_depth')
    depth_str = f"{depth}mm" if pd.notna(depth) else "未知"
    hist_detail = row.get('X_histology_detail', '')
    detail_str = f"（{hist_detail}）" if hist_detail and hist_detail != 'unknown' else ""

    desc = (f"{age}岁，{meno}，{hist}{detail_str}，{grade}，FIGO {stage}期，"
            f"肌层浸润{myo}，LVSI{lvsi}，盆腔淋巴结{lymph_pelvic}，"
            f"腹主动脉旁淋巴结{lymph_para}，宫颈受累{cervical}，"
            f"p53 {p53}，MMR {mmr}，分子分型{mol}，ESGO风险组{risk_display}。\n"
            f"合并症：{comorb_str}。附件受累{adnexal}，腹腔细胞学{cyto}，浸润深度{depth_str}。")
    return desc