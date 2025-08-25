from  zeroshot import embedding_zeroshot_ann


import pandas as pd

df = pd.read_csv("./testdata/amazon_massive_intent_zh-CN_train.csv")

candidate_labels = df["label_text"].unique().tolist()

print(f"candidate labels: {candidate_labels}")

# embedding_zeroshot_ann()
