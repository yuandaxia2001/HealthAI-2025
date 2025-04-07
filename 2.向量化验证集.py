import json
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

# 配置参数
API_KEY = "xxxx"  # 替换为您的API密钥
INPUT_FILE = '../official/official/20250208181531_camp_data_step_1_without_answer.jsonl'
OUTPUT_FILE = 'official_output_with_vectors.jsonl'
MAX_WORKERS = 50  # 根据API限制调整并发数
RETRY_COUNT = 3  # API调用重试次数
REQUEST_DELAY = 0.001  # 请求间隔（秒）

# 初始化客户端
client = ZhipuAI(api_key=API_KEY)


def get_embedding(text, retries=RETRY_COUNT):
    """调用Embedding-3模型并处理重试逻辑"""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="embedding-3",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            raise Exception(f"API调用失败: {str(e)}")
    return None


def process_line(line):
    """处理单行数据"""
    try:
        data = json.loads(line.strip())
        if 'feature_content' not in data:
            return None

        # 获取向量
        vector = get_embedding(data['feature_content'])

        # 合并结果
        return {
            **data,
            "feature_vector": vector
        }
    except Exception as e:
        print(f"处理ID {data.get('id', '未知')} 失败: {str(e)}")
        return None


def main():
    # 读取输入文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)  # 获取总行数用于进度条

    # 处理数据并保存
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            futures = [executor.submit(process_line, line) for line in fin]

            # 显示进度条
            with tqdm(total=total_lines, desc="处理进度") as pbar:
                for future in futures:
                    result = future.result()
                    if result:
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    pbar.update(1)
                    time.sleep(REQUEST_DELAY)  # 控制请求频率


if __name__ == "__main__":
    main()