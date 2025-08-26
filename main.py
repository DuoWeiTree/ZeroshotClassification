import csv
import llmclassify
import zeroshot
import numpy as np
import time
import re

def split_tags(tags: list[str]) -> list[str]:
	"""
	将包含逗号分隔标签的列表，分割成单个标签的列表。

	该函数遍历一个字符串列表，其中每个字符串可能包含一个或多个以逗号分隔的标签。
	它将所有标签分割成单独的字符串，并返回一个包含所有这些单个标签的新列表。

	Args:
		tags: 一个字符串列表，每个字符串包含一个或多个以逗号分隔的标签。

	Returns:
		一个包含所有单个标签的列表。

	Example:
		>>> split_tags(['a,b', 'c', 'd,e,f'])
		['a', 'b', 'c', 'd', 'e', 'f']
	"""
	result = []
	for tag in tags:
		result.extend(re.split(r"[,/]+", tag))
	return result

def mapping_tags(tags: list[str], map: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
	"""
	根据给定的映射字典，将标签列表转换为对应的标签、ID和分数列表。

	该函数遍历一个包含逗号分隔标签的列表。对于每个标签，它会查找映射字典。
	如果找到匹配项，则提取对应的标签名称、ID和分数。如果未找到，则用默认值填充。
	最后，将结果以逗号分隔的字符串形式返回。

	Args:
		tags: 一个字符串列表，每个字符串包含一个或多个以逗号分隔的标签。
		mapping: 一个字典，键为标签字符串，值为包含 'label', 'label_id' 和 'score' 的字典。

	Returns:
		一个包含三个列表的元组，分别对应处理后的标签、标签ID和分数。
		每个列表中的元素都是以逗号分隔的字符串。
	
	Example:
		>>> tags = ['tagA,tagB', 'tagC']
		>>> mapping = {'tagA': {'label': 'labelA', 'label_id': 1, 'score': 0.9},
					   'tagB': {'label': 'labelB', 'label_id': 2, 'score': 0.8},
					   'tagD': {'label': 'labelD', 'label_id': 4, 'score': 0.7}}
		>>> labels, label_ids, scores = mapping_tags(tags, mapping)
		>>> labels
		['labelA,labelB', 'None']
		>>> label_ids
		['1,2', '-1']
		>>> scores
		['0.9,0.8', '0']
	"""
	labels, label_ids, scores = [], [], []
	for tag in tags:
		label, label_id, score = [], [], []
		for single in re.split(r"[,/]+", tag):
			if single in map:
				output = map[single]
				label.append(output["label"])
				label_id.append(str(output["label_id"])) # 修复：确保转换为字符串
				score.append(str(output["score"])) # 修复：确保转换为字符串
			else:
				label.append("None")
				label_id.append("-1")
				score.append("0")
		labels.append(",".join(label))
		label_ids.append(",".join(label_id))
		scores.append(",".join(score))
	
	return labels, label_ids, scores

if __name__ == "__main__":
	# filename = "./tables/user_event_202508111636.csv"
	filename = "./tables/202508142031_产品信息.csv"
	
	column_name = "颜色"


	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		data = list(reader)
	
	sample_len = 9999999999
	data = data[:sample_len]

	sequences = [d[column_name] for d in data]
	sequences = np.array(sequences)

	# 排除空字符串
	not_empty = (sequences != "")

	sequences_single = split_tags(sequences[not_empty])

	extractor = llmclassify.CategoryExtractor("color_cache.json")
	# categories, _ = extractor.extend(sequences_single, column_name=column_name, N=200, min_N_total=1000, T=0.03)
	categories = extractor.get_categories()

	label_map = {
		c: i for i, c in enumerate(categories)
	}
	  
	print(f"Total {len(categories)} categories:\n{categories}")

	# 对非空行做Zero-shot分类

	classify_function = zeroshot.embedding_zeroshot_ann
	output_map = classify_function(sequences_single, categories, batch_size=1024)



	for key, output in output_map.items():
		output["label_id"] = label_map[output["label"]]


	labels, label_ids, scores = mapping_tags(sequences, output_map)


	# 给csv中的每一行做映射
	import pandas

	df = pandas.DataFrame()

	df["sequence"] = sequences
	df["labels"] = labels
	df["label_id"] = label_ids
	df["scores"] = scores
	
	import os
	os.makedirs("./labeled/", exist_ok=True)

	csv_name = f"./labeled/{column_name}-{time.time()}.csv"

	df.to_csv(csv_name)