import csv
import llmclassify
import zeroshot
import numpy as np
import time

if __name__ == "__main__":
	filename = "./tables/user_event_202508111636.csv"
	# filename = "./tables/202508142031_产品信息.csv"
	

	column_name = "summarize"

	with open(filename, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		data = list(reader)

	sample_len = 9999999999
	data = data[:sample_len]

	sequences = [d[column_name] for d in data]
	sequences = np.array(sequences)

	# 排除空字符串
	not_empty = (sequences != "")


	extractor = llmclassify.CategoryExtractor("cache.json")
	# categories, _ = extractor.extend(sequences[not_empty], column_name=column_name, N=200, min_N_total=1000, T=0.03)
	categories = extractor.get_categories()

	label_map = {
		c: i for i, c in enumerate(categories)
	}

	print(f"Total {len(categories)} categories:\n{categories}")

	# 对非空行做Zero-shot分类

	classify_function = zeroshot.embedding_zeroshot_ann
	output_map = classify_function(sequences[not_empty].tolist(), categories, batch_size=1024)

	
	"""
	with open("output", "r", encoding="utf-8") as file:
		import json
		output = json.load(fp=file)
		output_map = {}
		for o in output:
			sequence = o["sequence"]
			label = o["labels"][0]
			score = o["scores"][0]
			output_map[sequence] = {
				"label": label,
				"score": score
			}
	"""

	# 给csv中的每一行做映射
	results = []
	for d in data:
		sequence = d[column_name]
		if sequence not in output_map:
			d["label"] = "None"
			d["label_id"] = -1
			d["score"] = 0
		else:
			output = output_map[sequence]
			d["label"] = output["label"]
			d["label_id"] = label_map[output["label"]]
			d["score"] = output["score"]
		result = {
			column_name: sequence,
			"label": d["label"],
			"label_id": d["label_id"],
			"score": d["score"]
		}
		results.append(result)
	
	import os
	os.makedirs("./labeled/", exist_ok=True)

	with open(f"./labeled/{column_name}-{time.time()}.csv", "w", encoding="utf-8") as f:
		writer = csv.DictWriter(f=f, fieldnames=[key for key, _ in results[0].items()])
		writer.writeheader()
		writer.writerows(results)
