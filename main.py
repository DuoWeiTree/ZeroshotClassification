import csv
import llmclassify
import zeroshot
import numpy as np
import time
import pandas as pd
import json
import re

if __name__ == "__main__":
	# 不同格式的提取器

	split_single  = lambda seq : [seq]
	split_json_feature_name = lambda seq: [prt["featrue_name"] for prt in json.loads(seq)]
	split_json_name = lambda seq: [prt["name"] for prt in json.loads(seq)]
	split_json_segmented_users = lambda seq: [prt["segmented_users"] for prt in json.loads(seq)]
	split_list = lambda seq: seq.strip("{}'").split("','")


	# filename = "./tables/user_event_202508111636.csv"
	filename = "./tables/202508261024_UP表.csv"
	eid = "eid"
	column_name = "job"
	splitter = split_single

	cache_path = f"./cache/UP-{column_name}-cache.json"

	df = pd.read_csv(filename, keep_default_na=False, na_values=[])

	wsd = llmclassify.WSD(cache_path)
	
	wsd.extend(
		df=df, 
		column_name=column_name,
		splitter=splitter,
		N=200,
		min_N_total=1000,
		T=0.03,
		max_iteration=5
	)
	
	result = wsd.classify(
		df=df, 
		column_name=column_name,
		eid=eid,
		splitter= splitter,
	)

	output_path = f"./labeled/{column_name}-{time.time()}.csv"
	result.to_csv(output_path)