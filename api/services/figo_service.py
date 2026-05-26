import re
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# ─── 系统提示词 ────────────────────────────────────────────────────────────────

INSTRUCTION_FIGO = """你是一个专门从事妇科肿瘤分期和病理分析的医学专家系统。该系统根据FIGO指南对患者进行分期，并根据提供的病理报告给出合理的FIGO分期。
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
1.当报告没有提到相关特征时，不要随便推断，当作"该特征 = 无"处理！！！（尤其是肿瘤累及宫颈、卵巢、输卵管这些特征）
2.必须满足一条规则中的所有指标才能确认是该分期，只要有一项指标匹配不上就不是该分期！！！
3.你经常会忽略某些关键特征或者在判断时忘记某些指标，在输出最后分期前一定要确认是否遗漏！！！
4.你经常会将IIC期和IC期混淆，关键在于组织学肌层浸润深度是否存在肌层浸润（浸润浅肌层属于有肌层浸润！！！不属于"局限于息肉或局限于子宫内膜"！！！）（可以参考示例1、2）
5.当报告中关于盆腔淋巴结转移和主动脉旁淋巴结转移的说明未明确微/宏转移时，不要随便推断，按照"存在盆腔/主动脉旁淋巴结转移（未明确说明微/宏转移）"处理！！！但如果明确说明了，就一定要按微/宏转移处理！！！（可以参考示例3、4、5、6）
6.当初步判断为IIIC1期和IIIC2期时，一定要检查报告中是否明确了盆腔/主动脉淋巴结微/宏转移，防止出现遗漏！！！
7.你经常会犯将IIIA1期错误判断为其他分期的错误，关键在于肿瘤是否累及卵巢或输卵管，只要累及，必然是IIIA1期或其之后的分期！！！（可以参考示例7）

# 步骤2：分子分型决策
## 该步骤中需要的特征
1. 步骤1中初步判断出的FIGO分期
2.分子分型（病理报告）

## 规则
1. 分子分型不可用 => 保持原分期
2. 分子分型 = MMRd => 在分期后添加"mMMRd"作为下标。
3. 分子分型 = NSMP => 在分期后添加"NSMP"作为下标。
4. 分子分型 = POLEmut && FIGO分期 = IA1-IIC（包括IIC） => FIGO分期修改为IAmPOLEmut期。（重点关注！！！你经常忘记这条）
5. 分子分型 = POLEmut && FIGO分期 = IIIA1-IVC => 在分期后添加"POLEmut"作为下标。
6. 分子分型 = p53abn && FIGO分期 = IA2-IB,IIA-IIC && 组织学肌层浸润深度 = 存在肌层浸润 => FIGO分期修改为IICmp53abn期。
7. 分子分型 = p53abn && FIGO分期 = IA1,IC,IIIA1-IVC 或 FIGO分期 = IA3,IIA,IIB 局限于息肉或局限于子宫内膜 => 在分期后添加"p53abn"作为下标。

##重点注意！！！
1.当报告中没有提到分子分型时，不要随便推断，当作分子分型不可用处理！！！
2.当分子分型 = POLEmut 时，只要初步的FIGO分期判断在IIC（包括IIC）之前，最终FIGO分期都要修改为IAmPOLEmut期！！！

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

# 输出格式要求（非常重要！）
请在完成上述推理后，在最后一行单独输出：**FIGO分期：具体分期**
具体分期必须符合上述规则中的分期名称，例如：IA1期、IIIC1期、IAmPOLEmut期等。
不要输出其他无关文字。请务必遵守！

下面是一名患者的病理报告，请根据该报告进行诊断，判断出正确的FIGO分期：
"""

# ─── 正则：从模型输出中提取分期 ──────────────────────────────────────────────────
# 匹配形如 IAmPOLEmut、IIIC1ii、IVA、IB 等

_STAGE_PATTERN = re.compile(
    r"FIGO分期[：:]\s*"
    r"(I{1,3}(?:[ACV](?:[1-3](?:i{1,2})?)?|[VX])?(?:m(?:POLEmut|MMRd|p53abn)|(?:POLEmut|MMRd|NSMP|p53abn))?)",
    re.IGNORECASE,
)

_STAGE_FALLBACK = re.compile(
    r"\b(I(?:Am(?:POLEmut)|[ABC][1-3]?(?:i{1,2})?|[IVX]+[ABC]?[1-3]?(?:i{1,2})?)"
    r"(?:m(?:POLEmut|MMRd|p53abn)|(?:POLEmut|MMRd|NSMP|p53abn))?)\b",
)


def _extract_stage(raw_output: str) -> Optional[str]:
    """从模型原始输出中提取 FIGO 分期字符串，找不到则返回 None。"""
    m = _STAGE_PATTERN.search(raw_output)
    if m:
        return m.group(1).strip()
    # 次级匹配：找最后一个出现的分期标记
    matches = _STAGE_FALLBACK.findall(raw_output)
    if matches:
        return matches[-1].strip()
    return None


# ─── Service 类 ───────────────────────────────────────────────────────────────

class FigoService:
    """
    调用 vLLM（OriClinical / PathoLLM）进行 FIGO 分期推断。

    双模型策略：
      - Primary:  PathoLLM vLLM (OriClinical)，本地部署、领域专属
      - Fallback: DeepSeek API，当 Primary 不可用时自动回退

    使用 LangChain ChatOpenAI 异步接口，单例由 ResourceManager 维护。
    """

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "",
        fallback_enabled: bool = True,
        fallback_api_key: str = "",
        fallback_base_url: str = "",
        fallback_model_name: str = "",
    ) -> None:
        self._llm: Optional[ChatOpenAI] = None          # Primary (PathoLLM)
        self._fallback_llm: Optional[ChatOpenAI] = None  # Fallback (DeepSeek)
        self._base_url = base_url
        self._model_name = model_name
        self._fallback_enabled = fallback_enabled
        self._fallback_api_key = fallback_api_key
        self._fallback_base_url = fallback_base_url
        self._fallback_model_name = fallback_model_name

    def initialize(self) -> None:
        """同步初始化（在 ResourceManager.initialize 中调用）。"""
        self._llm = ChatOpenAI(
            model=self._model_name,
            base_url=self._base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=1024,
            request_timeout=600.0,
        )
        logger.info("FigoService [Primary] initialized: model=%s base_url=%s", self._model_name, self._base_url)

        # 初始化回退模型
        if self._fallback_enabled and self._fallback_api_key:
            self._fallback_llm = ChatOpenAI(
                model=self._fallback_model_name,
                base_url=self._fallback_base_url,
                api_key=self._fallback_api_key,
                temperature=0.0,
                max_tokens=1024,
                request_timeout=120.0,
            )
            logger.info(
                "FigoService [Fallback] initialized: model=%s base_url=%s",
                self._fallback_model_name, self._fallback_base_url,
            )
        elif self._fallback_enabled:
            logger.warning("FigoService: fallback 已启用但未配置 API key，回退功能不可用")
        else:
            logger.info("FigoService: fallback 已禁用")

    async def predict(self, patient_case: str) -> dict:
        """
        调用 LLM 进行 FIGO 分期，优先使用 Primary (PathoLLM vLLM)，
        连接失败时自动回退到 Fallback (DeepSeek API)。

        返回 {"figo_stage": str, "raw_output": str, "model_used": str}。
        若无法提取分期则 figo_stage 为 None。
        """
        if self._llm is None:
            raise RuntimeError("FigoService 尚未初始化，请先调用 initialize()")

        messages = [
            SystemMessage(content=INSTRUCTION_FIGO),
            HumanMessage(content=patient_case),
        ]

        raw_output: str = ""
        model_used: str = ""

        # ── Step 1: 尝试 Primary (PathoLLM vLLM) ──
        try:
            response = await self._llm.ainvoke(messages)
            raw_output = response.content if isinstance(response.content, str) else str(response.content)
            model_used = self._model_name
            logger.info("FigoService: Primary (PathoLLM) 调用成功 → %s", self._model_name)
        except Exception as exc:
            logger.warning("FigoService: Primary (PathoLLM) 调用失败: %s", exc)

            # ── Step 2: 回退到 Fallback (DeepSeek) ──
            if self._fallback_llm is not None:
                logger.info("FigoService: 正在回退到 Fallback (DeepSeek)...")
                try:
                    response = await self._fallback_llm.ainvoke(messages)
                    raw_output = response.content if isinstance(response.content, str) else str(response.content)
                    model_used = self._fallback_model_name
                    logger.info("FigoService: Fallback (DeepSeek) 调用成功 → %s", self._fallback_model_name)
                except Exception as fallback_exc:
                    logger.error("FigoService: Fallback (DeepSeek) 也失败了: %s", fallback_exc, exc_info=True)
                    raise RuntimeError(
                        f"FIGO 分期失败 — Primary ({self._model_name}) 和 Fallback ({self._fallback_model_name}) 均不可用"
                    ) from fallback_exc
            else:
                logger.error("FigoService: Fallback 未配置，无法回退")
                raise

        figo_stage = _extract_stage(raw_output)

        if figo_stage is None:
            logger.warning("FigoService: 未能从模型输出中提取分期。model=%s raw_output[:200]=%s", model_used, raw_output[:200])

        return {"figo_stage": figo_stage, "raw_output": raw_output, "model_used": model_used}
