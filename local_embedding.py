####
import requests
import json
import numpy as np

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