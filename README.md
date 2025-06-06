# HealthAI 2025 临床医学数据处理

比赛官网[HealthAI 2025 临床医学数据处理](https://competition.pkucxpl.com/competitions/7)

## 蒸馏deepseek r1并微调qwen2.5 instruct 7b模型

**目标：**

将deepseek r1的推理能力迁移到qwen2.5 instruct 7b中，使其在诊断医学病例时能够具备思考能力，提升诊断的结果，并可以严格按照要求的格式输出。

**整体思路：**

将低质、非格式化数据集构造为格式化数据集，再利用向量相似度，筛选出与目标数据最接近的高质数据。将高质数据输入给deepseek r1，保存其推理过程和输出结果，将其推理过程和输出结果制作成数据集，使用lora方式微调qwen2.5 instruct 7b。

### 原始数据集

使用Huatuo-26M 中文医疗问答数据集作为原始数据，介绍如下：

[Huatuo-26M 中文医疗问答数据集介绍 - 知乎](https://zhuanlan.zhihu.com/p/684831046)

使用了其中的精简版本Huatuo-Lite，可以在huggingface官网获取。[ Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)

这个数据集是网络上的医学问答内容，分为患者提问和医生回答。

### 构造医学垂直领域数据集

使用GLM-4-Plus模型，对原来的数据集进行改写，改写成比赛对应的数据格式。一共有17.8万条数据，多线程调用API容易出错、中断，因此调用批量推理进行推理。结果示例：

原数据：

```json
{
    "question": "我现在怀孕36周了自从怀孕以后血压就开始升高，以前一直都是110的70，最近这几次都是160.90，我很担心啊，我去医院检查了一下，医生说我是妊高症，我也不清楚到底怎么回事，不知道会不会影响孩子！",
    "origin_answer": "根据你的描述，你是妊高症，也就是妊娠期合并高血压综合征。一般情况下，你需要在绝对安静的环境中休息，减少疲劳，并按时服用医生开的降压药物。妊高症对胎儿的影响通常不会太大，但仍然需要密切关注。"
}
```

改写成竞赛数据集格式：

```json
{
    "feature_content": "性别: 女\n年龄: 未知\n主诉: 怀孕36周，血压升高\n现病史: 怀孕期间血压由110/70升高至160/90，诊断为妊高症\n既往史: 未知\n个人史: 未知\n过敏史: 未知\n婚育史: 怀孕36周\n体格检查: 血压升高 (160/90)"
}
```

### 向量化

将验证集（即官方给出的3000条数据）使用GLM-Embedding-3进行向量化。作为目标数据分布，再将由Huotuo-Lite构造而来的17.8万条数据向量化。由此就得到了3k个向量和17.8w个向量。这17.8w个构造而来的数据有些与目标数据接近，而有些与目标数据不相关。为了更好的微调模型，需要对17.8w条数据进行筛选。

### 利用向量相似度筛选优质数据

具体方法如下：

1. 将17.8w个向量中的每一个与官方的3k个向量做相似度匹配，然后选出top5
2. 将top5的得分求算术平均，再按该算数平均值排序
3. 选出前2w条，作为最终蒸馏deepseek r1时使用的数据集

下面是一个示例，question和origin_answer是huatuo数据集中的，feature_content是由华佗数据集构造而来的格式化数据，matches是官方的3k条数据集中，与该构造数据最匹配的5个。

```json
{
    "feature_content": "性别: 女\n年龄: 未提及\n主诉: 月经前乳房疼痛\n现病史: 近两个月来月经前四天乳房疼痛，月经来或快结束时疼痛缓解\n既往史: 未提及\n个人史: 未提及\n过敏史: 未提及\n婚育史: 未提及\n体格检查: 未提及",
    "question": "您好请问！我这两个月来月经前四天乳房疼痛，在月经来了，或者快走就不痛了，请问这是正常现象吗！经前乳房疼痛",
    "origin_answer": "经前乳房疼痛是正常的现象。在月经来临前的几天，女性体内的激素水平会升高，这些激素的刺激会导致乳房疼痛的感觉。一旦月经开始或者进行一些活动，乳房疼痛通常会减轻或消失。所以，您不用担心，这是正常的生理现象。如果疼痛严重或持续时间较长，建议您咨询医生以获取进一步的建议。",
    "matches": [
        {
            "id": 1288,
            "feature_content": "性别: 女\n年龄: 46\n主诉: 乳房疼痛持续时间时间单位\n现病史: 患者乳房疼痛，月经前疼痛明显。；否认慢性病疾病史，无吸烟史。\n既往史: -\n个人史: \n过敏史: \n婚育史: \n体格检查: 双乳未及明显肿块，双乳外上压痛，双侧腋窝未及肿大淋巴结。",
            "match_score": 0.8005416393280029
        },
        {
            "id": 469,
            "feature_content": "性别: 女\n年龄: 45\n主诉: 双乳疼痛3月\n现病史: 缘于3月前月经前出现双乳疼痛，以触痛明显。无乳头溢液。无发热。未曾诊治。随时间推移无明显加重。\n既往史: \n个人史: \n过敏史: \n婚育史: ",
            "match_score": 0.7446117401123047
        },
        {
            "id": 46,
            "feature_content": "性别: 男\n年龄: 34\n主诉: 右乳疼痛数天。\n现病史: 缘于数天前月经前出现右乳疼痛，以触痛明显。无乳头溢液。无发热。未曾诊治。随时间推移无明显加重。\n既往史: 既往体健。\n个人史: \n过敏史: \n婚育史: ",
            "match_score": 0.7081177234649658
        },
        {
            "id": 148,
            "feature_content": "性别: 女\n年龄: 41\n主诉: 主诉：乳房疼痛6月\n现病史: -；否认慢性病疾病史。\n既往史: -\n个人史: \n过敏史: \n婚育史: \n体格检查: ",
            "match_score": 0.702110230922699
        },
        {
            "id": 2009,
            "feature_content": "性别: 女\n年龄: 36\n主诉: 双乳疼痛1年。\n现病史: 缘于1年前月经前出现双乳疼痛，以触痛明显。无乳头溢液。无发热。未曾诊治。随时间推移无明显加重。无发热、乏力；无咳嗽，咳痰。无鼻塞、流涕、咽痛、肌痛；无腹泻，无胸闷、呼吸困难。\n既往史: 既往体健，无“高血压”，“糖尿病”，“冠心病”病史。否认“肝炎”、“结核”、“伤寒”、“痢疾”等传染病史。无手术史，无外伤史，无输血史，预防接种随当地进行。无药物、食物过敏史。无接触新冠肺炎患者；居住社区或曾出入社区内无新冠肺炎患者；工作或生活范围内无聚集性发病。\n个人史: \n过敏史: \n婚育史: ",
            "match_score": 0.6732081770896912
        }
    ]
}
```

### 调用deepseek r1生成推理过程和输出结果

将feature_content作为deepseek r1的唯一user prompt，并设置合适的system prompt，让deepseek r1诊断疾病，一共有2w条数据，为了节约成本和保证稳定，依然使用批量推理。

```python
system_prompt ='''你是一名资深全科医生，请严格按以下要求分析病例：
1. 诊断疾病（如果有多个疾病，分点列出）
2. 详细说明诊断依据（分点说明，每条依据用\\n分隔）
返回严格符合以下JSON格式：
{
    "reason": "诊断原因分点说明", 
    "diseases": "疾病名称"（如果有多种疾病，分点说明）
}
注意：
- 疾病名称使用标准医学术语
- 诊断原因需结合病例中的症状、体征、检查结果等要素
- 必须使用双引号，严格避免JSON格式错误
- 主诉中提到的所有疾病都要列出
- diseases的推导一定要依赖于reason的逻辑
明确结构化输出要求：
强制分点编号（如 “1. 2. 3.”）和医学逻辑顺序（主诉→现病史→体格检查→辅助检查→诊断依据）。
术语和格式一致性：
要求生成文本直接引用输入中的 原始术语（如 “无咳嗽咳痰”、“36.08℃”），避免同义替换。
覆盖关键信息：
强制模型覆盖输入中所有字段（主诉、现病史、体格检查等），避免遗漏。
限制生成自由度：
禁止添加输入中未提及的症状或检查建议，避免冗余内容降低精确度。'''
```

### 制作微调数据集并微调

使用llama facotry进行微调，按照要求构造数据集，将推理内容作为输出json中的第一项，然后跟着的是reason和diseases（diseases和reason的顺序很重要，曾试过将diseases放在reason的前面，在调用没有微调的qwen时，竟然都会输出乱码！），示例如下：
```json
  {
    "conversations": [
      {
        "from": "human",
        "value": "性别: 男\n年龄: 未提供\n主诉: 打喷嚏、鼻塞、鼻痒\n现病史: 患者近期出现打喷嚏、鼻塞和鼻痒等症状，无感冒史\n既往史: 未提供\n个人史: 未提供\n过敏史: 可能存在过敏体质\n婚育史: 未提供\n体格检查: 医生诊断结果为过敏性鼻炎"
      },
      {
        "from": "gpt",
        "value": "{\"reasoning_content\": \"好的，我现在需要处理这个病例分析。首先，用户的主诉是打喷嚏、鼻塞和鼻痒，这三个症状都是常见的上呼吸道问题。接下来看现病史，患者近期出现这些症状，而且没有感冒史，这提示可能不是感染性的疾病，比如普通感冒或者流感。既往史、个人史和婚育史没有提供，可能对诊断影响不大，但过敏史提到可能存在过敏体质，这是一个重要的线索。\\n\\n体格检查部分医生已经诊断了过敏性鼻炎，所以需要确认这个诊断是否合理。过敏性鼻炎的典型症状确实包括打喷嚏、鼻塞和鼻痒，而且通常没有发热或其他感染迹象。患者没有感冒史，这排除了急性上呼吸道感染的可能。过敏体质的存在进一步支持了过敏性鼻炎的诊断，因为过敏原是引发该病的主要原因。\\n\\n另外，需要考虑是否有其他可能的疾病，比如血管运动性鼻炎或慢性鼻炎，但这些通常会有不同的诱因，比如温度变化或气味，而患者提到的是过敏体质，所以更倾向于过敏性鼻炎。没有提到其他症状如流脓涕、头痛，所以不太可能是鼻窦炎。所以综合来看，过敏性鼻炎是最可能的诊断，且医生已经通过体格检查确认了这一点。\\n\", \"reason\": \"1. 主诉为打喷嚏、鼻塞、鼻痒等典型鼻部过敏症状\\n2. 现病史显示症状近期发作且无感冒史，排除感染性鼻炎\\n3. 过敏史提示可能存在过敏体质这一危险因素\\n4. 体格检查明确指向鼻腔过敏反应\", \"diseases\": \"过敏性鼻炎\"}"
      }
    ],
    "system": "你是一名资深全科医生，请严格按以下要求分析病例：\n1. 诊断疾病（如果有多个疾病，分点列出）\n2. 详细说明诊断依据（分点说明，每条依据用\\n分隔）\n返回严格符合以下JSON格式：\n{\n    \"reasoning_content\": \"自然语言推理过程\",\n    \"reason\": \"诊断原因分点说明\", \n    \"diseases\": \"疾病名称\"（如果有多种疾病，分点说明）\n}\n注意：\n- 疾病名称使用标准医学术语\n- 诊断原因需结合病例中的症状、体征、检查结果等要素\n- 必须使用双引号，严格避免JSON格式错误\n- 主诉中提到的所有疾病都要列出\n- diseases的推导一定要依赖于reason的逻辑\n明确结构化输出要求：\n强制分点编号（如 “1. 2. 3.”）和医学逻辑顺序（主诉→现病史→体格检查→辅助检查→诊断依据）。\n术语和格式一致性：\n要求生成文本直接引用输入中的 原始术语（如 “无咳嗽咳痰”、“36.08℃”），避免同义替换。\n覆盖关键信息：\n强制模型覆盖输入中所有字段（主诉、现病史、体格检查等），避免遗漏。\n限制生成自由度：\n禁止添加输入中未提及的症状或检查建议，避免冗余内容降低精确度。"
  }
```

然后调用llama factory进行微调，训练3个epoch，参数基本都使用默认的，能不改的就没改。

结果示例：
```json
{
    "id": 3001,
    "feature_content": "性别: 女\n年龄: 65\n主诉: 痰中带血3天\n现病史: 痰中带血3天，偶有咳嗽咳痰，无胸痛，无胸闷，大小便未见明显异常\n既往史: 否认其他重大疾病史 ，否认药物过敏史。\n个人史: \n过敏史: 湖北人 无烟酒病史\n婚育史: \n流行病史: \n体格检查: T ℃，神志清，咽部略充血，扁桃体轻度肿大，无化脓，双肺呼吸音粗。\n辅助检查: ",
    "qwen_result": "{\"reasoning_content\": \"好的，我现在需要处理这个患者的病例，并按照用户的要求进行详细的分析。首先，患者是女性，65岁，主诉是痰中带血三天。我得先考虑可能引起咯血的原因。\\n\\n根据主诉，主要症状是痰中带血，持续了三天，偶尔有咳嗽咳痰。没有胸痛和胸闷，这排除了一些急性呼吸道的问题比如肺炎或者气胸的可能性。另外，大小便正常，所以消化道出血或其他系统问题暂时不考虑。\\n\\n接下来看既往史和个人史：患者否认重大疾病和其他过敏史，也没有烟酒嗜好，但湖北人的身份可能提示某些地域性因素？不过这里的信息有限，可能不是重点。体格检查显示体温正常，神志清楚，咽部稍红肿，扁桃体轻度肿大但没化脓，双肺呼吸音粗糙。辅助检查部分目前没有提供数据，所以只能依靠现有资料判断。\\n\\n可能的鉴别诊断包括上呼吸道感染引起的鼻后滴漏综合征导致的咯血，或者是慢性支气管炎急性发作。因为咽部充血和扁桃体肿大可能是炎症的表现，而呼吸音粗可能涉及下呼吸道。此外，老年患者出现反复咯血还要考虑结核复发或者肺癌的可能，但由于缺乏影像学证据，不能确定。\\n\\n诊断依据方面，首先是主诉的痰中带血，这是核心症状。其次是伴随的咳嗽咳痰，虽然次数不多，但仍存在呼吸道刺激的情况。咽部充血和扁桃体肿大会支持上呼吸道感染的存在，进而可能导致局部血管脆性增加而出血。同时，双肺呼吸音粗可能表明有轻微的支气管炎症反应，这也与感染有关。由于没有发热、胸痛等症状，更倾向于局限在上呼吸道或近端支气管的病变而非深部肺组织或全身性疾病。\\n\\n因此，在这些条件下最可能的两个诊断是急性上呼吸道感染和慢性支气管炎急性加重。需要进一步胸部X线或CT确认是否有肺部实质性的改变，以及血液检查评估感染指标是否升高。但当前可用的数据不足以做出更多结论，必须基于现有的临床表现来排序可能性最高的情况。\\n\", \"reason\": \"1. 主诉为痰中带血伴偶发咳嗽咳痰，符合呼吸道黏膜损伤导致毛细血管破裂出血的特点\\n2. 身体检查发现咽部略充血及扁桃体轻度肿大，提示上呼吸道感染灶\\n3. 双肺呼吸音粗反映可能存在支气管黏膜水肿或分泌物增多\\n4. 年龄65岁属高危人群，需优先排查恶性肿瘤可能\\n5. 缺乏发热、胸痛等特异性症状暂无法确诊严重器质性疾病\", \"diseases\": \"1. 急性上呼吸道感染\\n2. 慢性支气管炎急性发作\"}"
}
```

