import json
import re
from tqdm import tqdm


def process_data(requests_path, results_path, output_path):
    # 读取请求数据
    requests = {}
    with open(requests_path, 'r', encoding='utf-8') as f:
        for line in f:
            req = json.loads(line)
            requests[req["custom_id"]] = req["body"]["messages"]

    # 读取结果数据
    results = {}
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            res = json.loads(line)
            custom_id = res["custom_id"]
            content = res["response"]["body"]["choices"][0]["message"]["content"]
            reasoning = res["response"]["body"]["choices"][0]["message"].get("reasoning_content", "")

            # 解析原始JSON并添加推理内容
            try:
                structured_data = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())
                merged_json = {
                    "reasoning_content": reasoning,
                    "reason": structured_data.get("reason", ""),
                    "diseases": structured_data.get("diseases", "")
                }
                results[custom_id] = json.dumps(merged_json, ensure_ascii=False)
            except Exception as e:
                print(f"解析错误 {custom_id}: {str(e)}")
                continue

    # 构建sharegpt格式
    formatted_data = []
    for custom_id in tqdm(requests):
        messages = requests[custom_id]
        response = results.get(custom_id, "")

        # 跳过无效数据
        if not response:
            continue

        # 构建对话结构
        conversation = []
        system_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                # 更新系统提示包含新字段
                system_prompt = msg["content"].replace(
                    '"reason": "诊断原因分点说明"',
                    '"reasoning_content": "自然语言推理过程",\n    "reason": "诊断原因分点说明"'
                )
            elif msg["role"] == "user":
                conversation.append({
                    "from": "human",
                    "value": msg["content"]
                })

        conversation.append({
            "from": "gpt",
            "value": response  # 直接使用纯JSON
        })

        formatted_data.append({
            "conversations": conversation,
            "system": system_prompt,
            "tools": json.dumps({
                "output_format": {
                    "reasoning_content": "string",
                    "reason": "string",
                    "diseases": "string"
                }
            })
        })

    # 保存转换后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，有效数据{len(formatted_data)}条")


# 使用示例
# process_data(
#     requests_path="batch_requests.jsonl",
#     results_path="results (1).jsonl",
#     output_path="medical_reasoning.json"
# )

import json
import re


def check_dataset(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)

    error_log = []
    for idx, item in enumerate(data):
        try:
            response = item['conversations'][1]['value']
            data = json.loads(response)

            # 验证推理内容
            if len(data['reasoning_content']) < 150:
                error_log.append(f"样本{idx}: 推理内容过短")

            # 验证诊断依据格式
            if not re.match(r'^\d+\.\s', data['reason']):
                error_log.append(f"样本{idx}: 诊断依据格式错误")

        except Exception as e:
            error_log.append(f"样本{idx}: JSON解析失败 - {str(e)}")

    print(f"检测完成，发现{len(error_log)}个问题")
    return error_log

print(check_dataset('medical_reasoning.json'))