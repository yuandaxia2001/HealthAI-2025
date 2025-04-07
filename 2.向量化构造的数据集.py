import os
import json
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import time
# 配置参数
API_KEY = "xxx"  # 替换为实际API密钥
INPUT_FILE = 'result.jsonl'  # 输入文件路径
OUTPUT_FILE = 'huatuo18M_vectorized.jsonl'  # 输出文件路径
MAX_WORKERS = 10  # 并发线程数（根据API限制调整）

# 初始化客户端
client = ZhipuAI(api_key=API_KEY)


def process_entry(entry):
    """处理单个JSON条目，生成向量"""
    try:
        # 提取feature_content文本
        text = entry.get("feature_content", "")

        # 调用Embedding-3模型
        response = client.embeddings.create(
            model="embedding-3",
            input=text,
            timeout=10,
        )

        # 提取向量（假设响应结构符合预期）
        vector = response.data[0].embedding

        # 添加向量到条目
        entry["feature_vector"] = vector
        return entry
    except Exception as e:
        print(f"处理ID {entry.get('id', '未知')} 失败: {str(e)}")
        return None


def main():
    # 读取输入文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        entries = [json.loads(line) for line in f_in]

    num = 0
    # 多线程处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry) for entry in entries]

        # 写入结果（带进度条）
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="向量化处理"):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                # num += 1
                # if num%500==0:
                #     time.sleep(10)


if __name__ == "__main__":
    main()