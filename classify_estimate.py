from  zeroshot import embedding_zeroshot_ann

import pandas as pd

df = pd.read_csv("./testdata/amazon_massive_intent_zh_en.csv")

print(df.loc[0])
candidate_labels = df["label_text"].unique().tolist()

print(f"candidate labels: {candidate_labels}")

# embedding_zeroshot_ann()
covered = []
for _ in range(10):
    from llmclassify import WSD, Sampler

    wsd = WSD()
    sampler: Sampler
    _, _, sampler = wsd.extend(df=df, 
                            column_name="text", 
                            splitter=lambda seq: [seq], 
                            max_iteration=8,
                            return_sampler=True
                        )


    total_indices = list(range(len(df)))

    selected_indices = list(set(total_indices) - set(sampler.indices))
    covered_labels = df.loc[selected_indices, "label_text"].unique().tolist()
    
    covered.append(len(covered_labels))
    print(f"共 {len(candidate_labels)} 类, 覆盖 {len(covered_labels)} 类, 覆盖率 {len(covered_labels) / len(candidate_labels)}")


print(f"执行 {len(covered)} 次, 覆盖率分别为 {covered}")