import numpy as np
from FlagEmbedding import FlagModel
from numpy.linalg import norm
from typing import List, Tuple
from transformers import pipeline

####
import requests
import json

def l2_normalize(x):
	"""
	对一个(N, M)的向量x，在M维上进行L2归一化。
	"""
	# 计算M维上的L2范数
	# keepdims=True 保持维度，方便后续广播运算
	norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
	# 防止除以零
	norm[norm == 0] = 1e-12
	# 归一化
	return x / norm

def get_embedding(tags) -> list:
	url = 'http://192.168.66.201:8000/embedding'
	headers = {'Content-Type': 'application/json'}
	data = {'text': list(tags)}
	try:
		response = requests.post(url, headers=headers, data=json.dumps(data), timeout=300)
		response.raise_for_status()
		return response.json()["embeddings"]
	except requests.exceptions.RequestException as e:
		print(f"请求发生错误: {e}")
		return None

def encode_batch(tags, normalize_embeddings=True) -> np.ndarray:
	tags_unique = np.unique(tags)
	emb_unique = get_embedding(tags=tags_unique)
	emb_mapping = {
		tags_unique[i]: emb_unique[i]
		for i in range(len(tags_unique))
	}

	emb = [emb_mapping[t] for t in tags]
	emb = np.array(emb)
	if normalize_embeddings:
		magnitude = (emb ** 2).sum(axis=-1, keepdims=True)
		magnitude[magnitude == 0] = 1
		emb /= np.sqrt(magnitude)
	return emb

def encode(tags, batch_size=5000, **kwargs) -> np.ndarray:
	emb = []
	for i in range(0, len(tags), batch_size):
		emb += list(encode_batch(tags=tags[i: i + batch_size], **kwargs))
		
	return np.array(emb)

####

import faiss
from sentence_transformers import SentenceTransformer

import torch

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

def embedding_zeroshot_ann(sequences_to_classify: list[str], candidate_labels: list[str], batch_size: int = 512):
	"""
	使用嵌入模型和 Faiss 高效执行零样本分类。

	这个函数通过以下步骤完成分类：
	1. 将候选标签和待分类序列都编码为向量。
	2. 使用 Faiss 库对标签向量建立高效索引。
	3. 通过近似最近邻搜索（ANN），快速为每个序列找到最匹配的标签向量。

	参数:
	- sequences_to_classify (list[str]): 待分类的文本序列列表。
	- candidate_labels (list[str]): 候选类别标签列表。
	- batch_size (int, 可选): 批处理大小，用于加速模型编码。默认值为 1024。

	返回:
	- dict: 一个字典，键为原始序列，值为一个包含最佳匹配标签和对应分数的字典。
			例如: `{"sequence": {"label": "label_name", "score": 0.95}}`。
	"""
	# 1. 加载 FlagEmbedding 模型
	# 使用专门为检索任务优化的指令，以生成高质量的查询向量。
	# model = FlagModel('BAAI/bge-large-zh-v1.5', use_fp16=True)
	"""
	model = SentenceTransformer(
		"Qwen/Qwen3-Embedding-0.6B",
		model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
		tokenizer_kwargs={"padding_side": "left"},
	)
	"""
	# 记录开始时间，用于性能评估
	import time
	start_time = time.time()

	# 2. 对候选标签进行编码
	# 标签向量作为“文档”，无需使用查询指令。
	print("正在对候选标签进行编码...")
	label_embeddings = encode(candidate_labels, batch_size=batch_size)
	# model.encode(candidate_labels, batch_size=batch_size)

	# 4. 对待分类的序列进行编码
	# 序列作为“查询”，需要使用查询指令来优化生成的向量。
	print("正在对查询序列进行编码...")
	query_embeddings = encode(sequences_to_classify, batch_size=batch_size)
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
			"score": best_match_scores[i]
		}
	
	return mapping

def zeroshot_classify(sequences_to_classify: list[str], candidate_labels: list[str], batch_size: int = 512):
	"""
	使用 Hugging Face `pipeline` 进行零样本分类。

	这个函数利用预训练的 NLI（自然语言推理）模型，将待分类的序列和候选标签
	转换为一个假设（hypothesis），然后判断该假设是否与序列（前提，premise）
	相矛盾或蕴含，以此来确定最佳匹配的标签。

	参数:
	- sequences_to_classify (list[str]): 需要分类的句子或文本序列列表。
	- candidate_labels (list[str]): 候选类别标签列表。
	- batch_size (int, 可选): 批处理大小，用于加速推理。默认值为 1024。

	返回:
	- dict: 一个字典，键为原始序列，值为一个包含最佳匹配标签和对应分数
			的字典，例如: `{"sequence": {"label": "label_name", "score": 0.95}}`。
	"""
	# 使用 Hugging Face 的 `pipeline` API 加载零样本分类模型。
	# 该模型基于 NLI，能够判断文本之间的蕴含关系。
	# "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" 是一个多语言的 NLI 模型，
	# 适用于零样本分类任务。
	model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
	classifier = pipeline("zero-shot-classification", model=model_name)
	
	# 导入计时库，用于性能评估
	import time
	start_time = time.time()

	# 导入进度条库，用于可视化处理进度
	from tqdm import tqdm

	output = []

	# 使用批处理来处理大规模数据，可以显著提高处理速度
	for i in tqdm(range(0, len(sequences_to_classify), batch_size)):
		batch = sequences_to_classify[i:i+batch_size]
		
		# 调用 classifier 对一个批次进行分类。
		# - `multi_label=False`: 表示每个序列只应被分配一个最佳标签。
		# - `batch_size`: 将模型推理也进行批处理以提高效率。
		output_batch = classifier(batch, candidate_labels, multi_label=False, batch_size=batch_size)
		
		# 将当前批次的结果添加到总结果列表中
		output.extend(output_batch)

	# 结束计时
	end_time = time.time()
	time_cost = end_time - start_time
	print(f"Performed zero-shot classification on {len(sequences_to_classify)} sequences in {time_cost:.2f} seconds.")

	
	# 将分类结果保存到本地文件
	with open("output", "w", encoding="utf-8") as file:
		import json
		# 使用 `json.dumps` 格式化输出，使其更具可读性
		print(json.dumps(output, indent=4, ensure_ascii=False), file=file)
	

	# 创建一个更简洁、易于使用的映射字典
	mapping = {}
	for o in output:
		sequence = o["sequence"]
		label = o["labels"][0]  # 取第一个（即分数最高的）标签
		score = o["scores"][0]  # 取对应的分数
		mapping[sequence] = {
			"label": label,
			"score": score
		}
	
	return mapping


if __name__ == "__main__":
	sequences_to_clasisfy = ["小", "中", "大", "大号", "超大"]
	candidate_labels = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
	# sequences_to_clasisfy = ["1", "2", "3"]
	# candidate_labels = ["1", "2", "3"]
	print(embedding_zeroshot_ann(sequences_to_clasisfy, candidate_labels))
	