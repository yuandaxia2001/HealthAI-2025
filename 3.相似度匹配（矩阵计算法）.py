import json
import numpy as np
from tqdm import tqdm


def load_full_records(filename):
    """加载完整数据记录，保留所有原始字段"""
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"加载 {filename}"):
            item = json.loads(line)
            # 转换特征向量为numpy数组（如果存在）
            if 'feature_vector' in item:
                item['feature_vector'] = np.array(item['feature_vector'], dtype=np.float32)
            records.append(item)
    return records


def main():
    # 加载完整目标数据集（保留所有字段）
    print("1. 加载目标数据集...")
    target_data = load_full_records('official_output_with_vectors.jsonl')

    # 预处理目标数据
    target_ids = [item['id'] for item in target_data]
    target_vectors = np.array([item['feature_vector'] for item in target_data])
    target_norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    target_norms = np.where(target_norms == 0, 1e-10, target_norms)
    target_normalized = target_vectors / target_norms

    # 加载完整查询数据（保留所有字段）
    print("\n2. 加载查询数据集...")
    query_data = load_full_records('huatuo18M_vectorized.jsonl')

    # 内存优化分块处理
    BATCH_SIZE = 5000  # 根据内存容量调整
    total_batches = (len(query_data) + BATCH_SIZE - 1) // BATCH_SIZE

    with open('full_output_with_all_fields.jsonl', 'w', encoding='utf-8') as fout:
        for batch_idx in tqdm(range(total_batches), desc="处理进度"):
            # 获取当前批次数据
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, len(query_data))
            batch = query_data[start:end]

            # 构建查询矩阵
            query_vectors = np.array([item['feature_vector'] for item in batch])
            query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_norms = np.where(query_norms == 0, 1e-10, query_norms)
            query_normalized = query_vectors / query_norms

            # 矩阵加速计算相似度
            similarity_matrix = np.dot(query_normalized, target_normalized.T)

            # 高效获取Top5结果
            k = 5
            top_indices = np.argpartition(-similarity_matrix, k, axis=1)[:, :k]
            sorted_indices = np.argsort(-np.take_along_axis(similarity_matrix, top_indices, axis=1), axis=1)
            final_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)

            # 构建完整结果记录
            for i in range(len(batch)):
                # 深拷贝原始数据
                original_record = dict(batch[i])

                # 添加匹配结果（保留目标记录所有字段）
                original_record['matches'] = []
                for idx in final_indices[i]:
                    target_record = dict(target_data[idx])  # 复制目标记录
                    # 移除目标记录的特征向量
                    if 'feature_vector' in target_record:
                        del target_record['feature_vector']
                    # 添加相似度分数
                    target_record['match_score'] = float(similarity_matrix[i, idx])
                    original_record['matches'].append(target_record)

                # 移除查询记录的特征向量
                if 'feature_vector' in original_record:
                    del original_record['feature_vector']

                # 写入结果
                fout.write(json.dumps(original_record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()