import json
from collections import defaultdict
from tqdm import tqdm
def process_file(input_file, output_file):
    # 读取并处理数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="正在加载数据"):
            try:
                item = json.loads(line)
                # 计算平均分
                if 'matches' in item and len(item['matches']) > 0:
                    scores = [match.get('match_score', 0) for match in item['matches']]
                    item['avg_score'] = sum(scores) / len(scores)
                else:
                    item['avg_score'] = 0  # 处理无匹配的情况
                data.append(item)
            except json.JSONDecodeError:
                print(f"解析错误: {line}")
                continue

    # 按平均分排序
    sorted_data = sorted(data,
                        key=lambda x: x['avg_score'],
                        reverse=True)

    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(sorted_data, desc="正在保存结果"):
            # 移除临时添加的avg_score字段
            if 'avg_score' in item:
                del item['avg_score']
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



system_prompt = '''你是一名资深全科医生，请严格按以下要求分析病例：
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


def convert_to_batch():
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析原始数据
            data = json.loads(line.strip())

            # 构建请求体
            request = {
                "custom_id": str(data["id"]),
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": data["feature_content"]
                        }
                    ],
                    "max_tokens": 5000,
                    #"top_p": 1
                }
            }

            # 写入JSONL文件
            outfile.write(json.dumps(request, ensure_ascii=False) + '\n')



if __name__ == "__main__":

    ## 排序
    # input_path = "full_output_with_all_fields.jsonl"    # 输入文件路径
    # output_path = "sorted_output.jsonl" # 输出文件路径
    # process_file(input_path, output_path)
    #

    # ## 取两万行
    input_file = 'sorted_output.jsonl'  # 输入文件路径
    output_file = '2w_data.jsonl'  # 输出文件路径
    max_lines = 20000  # 最大提取行数

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for line in infile:
            if count >= max_lines:
                break
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                # 提取需要的字段
                extracted = {
                    "feature_content": data.get("feature_content"),
                    "id": data.get("id")
                }
                # 写入输出文件
                json.dump(extracted, outfile, ensure_ascii=False)
                outfile.write('\n')
                count += 1
            except json.JSONDecodeError:
                print(f"JSON解析失败（第 {count + 1} 行附近）")
            except KeyError as e:
                print(f"缺失字段 {e}（第 {count + 1} 行）")

    print(f"处理完成，共提取 {count} 行数据。")


    ## 生成批量处理的文件
    # 配置参数
    input_file = '2w_data.jsonl'  # 之前生成的feature_content数据文件
    output_file = 'batch_requests.jsonl'


    convert_to_batch()
    print(f"批量推理文件已生成：{output_file}")