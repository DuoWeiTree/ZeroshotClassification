
from zeroshot import embedding_zeroshot_ann
from local_embedding import encode
import json
import pandas as pd
from itertools import chain

with open("./testdata/colors_synonyms.json", "r", encoding="utf-8") as file:
    colors = json.load(fp=file)



candidate_labels = [key for key, value in colors.items()]

sequences = list(chain(*[value for key, value in colors.items()]))
output_map = embedding_zeroshot_ann(encoder=encode, sequences_to_classify=sequences, candidate_labels=candidate_labels)

total = 0
correct = 0
for seq in sequences:
    total += 1
    belong = output_map[seq]["label"]
    if seq in colors[belong]:
        correct += 1

print(f"total: {total}, correct: {correct}, acc: {correct / total}")