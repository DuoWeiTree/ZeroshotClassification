import numpy as np
from typing import Callable, Iterable
import faiss

def ann_similarity(query_embeddings: np.ndarray, label_embeddings: np.ndarray):

	label_embeddings = label_embeddings.astype('float32')
	query_embeddings = query_embeddings.astype('float32')

	# 3. 创建 Faiss 索引
	# 向量维度，例如 BGE-large 模型为 1024
	d = label_embeddings.shape[1]
	
	# 使用 `IndexFlatIP`（内积索引），因为归一化后的向量内积等同于余弦相似度，
	# 且能提供高效的搜索。
	index = faiss.IndexFlatIP(d)
	
	# 将所有标签向量添加到索引中，以便进行快速搜索
	index.add(label_embeddings)


	# 5. 使用 Faiss 执行高效搜索 (ANN)
	# 查找每个查询向量在索引中最接近的 1 个邻居。
	print("正在使用 Faiss 进行搜索...")
	k = 1
	distances, indices = index.search(query_embeddings, k)
	# Faiss 返回的 `distances` 即为相似度分数（内积）
	best_match_scores = distances.flatten()
	best_matches_indices = indices.flatten()

	return best_matches_indices, best_match_scores

def embedding_zeroshot_ann(
	encoder: Callable[[list[str]], Iterable[Iterable[float]]],
	sequences_to_classify: list[str],
	candidate_labels: list[str],
	batch_size: int = 512,
) -> dict[str, dict[str, ]]:
	"""
	利用嵌入模型和近似最近邻（ANN）方法执行零样本分类。

	该函数通过以下步骤完成分类：
	1. 使用 `encoder` 将候选标签和待分类序列编码为向量。
	2. 基于标签向量构建高效的 ANN 索引。
	3. 对查询序列执行最近邻搜索，找到最匹配的标签及其相似度分数。

	参数:
	- encoder (Callable[[list[str]], Iterable[Iterable[float]]]): 将字符串列表编码为向量表示的函数。
	- sequences_to_classify (list[str]): 待分类文本序列。
	- candidate_labels (list[str]): 候选标签。
	- batch_size (int, 可选): 批处理大小，用于加速编码过程。默认值为 512。

	返回:
	- dict[str, dict[str, float]]: 结果字典。键为原始输入序列，值为一个字典，
	  其中包含：
		- "label" (str): 最佳匹配的标签。
		- "score" (float): 匹配的相似度分数。

	示例:
	>>> encoder = lambda texts, batch_size=512: some_model.encode(texts)
	>>> sequences = ["我想买手机", "推荐一部小说"]
	>>> labels = ["电子产品", "文学", "体育"]
	>>> embedding_zeroshot_ann(encoder, sequences, labels)
	{
		"我想买手机": {"label": "电子产品", "score": 0.93},
		"推荐一部小说": {"label": "文学", "score": 0.89}
	}
	"""
	# 记录开始时间，用于性能评估
	import time
	start_time = time.time()

	# 2. 对候选标签进行编码
	# 标签向量作为“文档”，无需使用查询指令。
	print("正在对候选标签进行编码...")
	label_embeddings = encoder(candidate_labels, batch_size=batch_size)
	# model.encode(candidate_labels, batch_size=batch_size)

	# 4. 对待分类的序列进行编码
	# 序列作为“查询”，需要使用查询指令来优化生成的向量。
	print("正在对查询序列进行编码...")
	query_embeddings = encoder(sequences_to_classify, batch_size=batch_size)
	# model.encode(sequences_to_classify, batch_size=batch_size)

	# best_matches_indices, best_match_scores = ann_similarity(query_embeddings, label_embeddings)
	# similarity = model.similarity(query_embeddings, label_embeddings).numpy()
	best_matches_indices, best_match_scores = ann_similarity(query_embeddings, label_embeddings)
	# best_matches_indices = similarity.argmax(axis=1)
	# best_match_scores = similarity.max(axis=1)

	# 记录结束时间，并打印总耗时
	end_time = time.time()
	time_cost = end_time - start_time
	print(f"Performed embedding-zero-shot classification on {len(sequences_to_classify)} sequences in {time_cost:.2f} seconds.")


	# 6. 将索引转换为标签，并构建结果字典
	predicted_labels = [candidate_labels[i] for i in best_matches_indices]

	mapping = {}
	for i, sequence in enumerate(sequences_to_classify):
		mapping[sequence] = {
			"label": predicted_labels[i],
			"label_id": best_matches_indices[i],
			"score": best_match_scores[i]
		}
	
	return mapping
