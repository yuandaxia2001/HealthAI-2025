import os
import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm

# 配置参数
API_KEY = "xxx"  # 替换为你的API密钥
INPUT_FILE = 'huatuo18M_test.jsonl'
BATCH_INPUT_FILE = 'huatuo_batch_input.jsonl'
BATCH_OUTPUT_FILE = 'huatuo_batch_output.jsonl'
FINAL_OUTPUT_FILE = 'huatuo18M_processed.jsonl'

# 初始化客户端
client = ZhipuAI(api_key=API_KEY)

prompt = \
    '''
    你是一名专业的医疗助手，接下来会给你一系列数据，每一项数据均由以下几个部分组成：
    1. 患者提问
    2. 医生回答
    3. 相关疾病
    
    请仔细阅读患者提问和医生回答，分析其中的因果关系并给出诊断依据。不要虚构内容。请确保你的诊断依据清晰、有条理。你需要逐步推理，给出详细的推理过程。
    
    请按照以下步骤处理输入数据：
    
    1. 判断患者提问是否是在问诊，如果不是，则在返回结果的consult字段中填入"false"并返回，其余字段为空字符串。
        - 如果患者在询问得了什么病，则判断为问诊。
        - 如果患者在询问科普性质的问题，则判断为非问诊。
    
    2. 仔细分析患者的问题，提取患者的主诉、检查结果和病史：
        - 提取患者的人口统计学信息（性别、年龄）
        - 识别主要症状和持续时间，提取现病史
        - 归纳病人的主诉
        - 若病人提及，整理病人的既往史、个人史、婚育史、过敏史、体格检查或辅助检查等
        - 对病例信息的完整程度进行打分，分数为0-5分，0分表示没有提供任何信息，5分表示提供了非常详细的信息
            * 若性别、年龄、主诉、现病史不完整，打分应该为0～2分
            * 若性别、年龄、主诉、现病史完整，根据其详细程度，以及是否给出既往史、个人史、婚育史、过敏史、体格检查或辅助检查进行进一步的打分
    
    
    3. 根据医生回答和相关疾病，给出疾病的诊断结果，并对诊断过程进行推理：
        - 将临床表现与病理特征对应
        - 分析主诉中与疾病相关的信息
        - 分析检查结果与疾病诊断的关系
        - 分析病史与疾病的关系
        - 你应该从医生的视角给出诊断过程和诊断依据，不要以第三人称描述
        - 诊断依据和推理过程不需要包含后续治疗建议
        - 其他可能的分析
    
    4. 按照指定格式组织输出：
        - 'consult'字段填入"true"表示是问诊，"false"表示不是问诊
        - 'reason'部分采用分点式医学推理
        - 'diseases'为诊断结果
        - 'feature_content'结构化患者病例信息
        - 'score'为病例信息的完整程度打分
    
    <患者提问>
    {question}
    </患者提问>
    
    <医生回答>
    {answer}
    </医生回答>
    
    <相关疾病>
    {disease}
    </相关疾病>
    
    你必须严格按照以下格式输出
    ```json
    {{
    "consult": 字段填入"true"表示是问诊，"false"表示不是问诊",
    "reason": "1. [推理过程][结合主诉、病史等的诊断推理过程]；\\n2. ...（按优先级排序）",
    "diseases": "[若有多项疾病，使用序号表示，如：1. [疾病1]；2. [疾病2]；...]",
    "feature_content": "性别: [提取值]\\n年龄: [提取值]\\n主诉: [归纳症状]\\n现病史: [整理病程]\\n既往史: [相关病史]\\n个人史: [相关病史]\\n过敏史: [相关病史]\\n婚育史: [相关病史]\\n体格检查: [相关检查结果] (若未提及相关病史，使用空字符串填充即可)",
    "score": "[0-5分，0分表示病例没有提供任何信息，5分表示提供了非常详细的信息]"
    }}
    ```
    '''
def build_batch_requests(input_file, batch_file):
    """构建符合Batch API要求的输入文件"""
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(batch_file, 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(tqdm(fin, desc='构建Batch请求')):
            data = json.loads(line)

            # 构造请求体
            request_body = {
                "model": "glm-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一名专业的医疗助手，请你阅读患者提问和医生回答，给出医生的诊断结果和诊断过程，以及结构化的患者病例信息。"
                    },
                    {
                        "role": "user",
                        "content": prompt.format(
                            question=data['question'],
                            answer=data['answer'],
                            disease=data.get('related_diseases', '')
                        )
                    }
                ],
                "temperature": 0.1,
                "top_p": 0.1
            }

            # 写入JSONL格式
            batch_request = {
                "custom_id": f"case_{data['id']}",  # 使用原始ID保证唯一性
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": request_body
            }
            fout.write(json.dumps(batch_request, ensure_ascii=False) + '\n')


def process_batch_results(batch_output_file, final_output_file):
    """处理Batch API返回结果"""
    results = {}

    # 读取原始数据建立ID映射
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        original_data = {json.loads(line)['id']: json.loads(line) for line in f}

    # 解析Batch结果
    with open(batch_output_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='处理Batch结果'):
            try:
                response = json.loads(line)
                custom_id = response['custom_id']

                # 提取有效响应
                if response['response']['status_code'] == 200:
                    content = response['response']['body']['choices'][0]['message']['content']

                    # 解析JSON内容
                    try:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)

                        # 合并原始数据
                        original = original_data.get(int(custom_id.split('_')[1]), {})
                        merged = {
                            **result,
                            "id": original.get('id'),
                            "label": original.get('label'),
                            "question": original.get('question'),
                            "origin_answer": original.get('answer')
                        }
                        results[custom_id] = merged
                    except Exception as e:
                        print(f"解析失败 {custom_id}: {str(e)}")

            except Exception as e:
                print(f"无效行: {str(e)}")

    # 写入最终结果
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for item in results.values():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # 步骤1：构建Batch请求文件
    build_batch_requests(INPUT_FILE, BATCH_INPUT_FILE)

    # 步骤2：上传文件
    print("上传Batch文件...")
    upload_response = client.files.create(
        file=open(BATCH_INPUT_FILE, "rb"),
        purpose="batch"
    )
    file_id = upload_response.id
    print(f"文件已上传，ID: {file_id}")

    # 步骤3：创建Batch任务
    print("创建Batch任务...")
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v4/chat/completions",
        auto_delete_input_file=True,
        metadata={"description": "Huatuo医疗数据处理"}
    )
    batch_id = batch_job.id
    print(f"Batch任务已创建，ID: {batch_id}")

    # 步骤4：监控任务状态
    while True:
        status = client.batches.retrieve(batch_id).status
        print(f"当前状态: {status}")
        if status == 'completed':
            break
        time.sleep(6)

    # 步骤5：下载结果
    print("下载结果文件...")
    batch_info = client.batches.retrieve(batch_id)

    # 下载成功结果
    success_content = client.files.content(batch_info.output_file_id)
    success_content.write_to_file(BATCH_OUTPUT_FILE)

    # 处理结果
    process_batch_results(BATCH_OUTPUT_FILE, FINAL_OUTPUT_FILE)

    print("处理完成！")


if __name__ == "__main__":
    main()