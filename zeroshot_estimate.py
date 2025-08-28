
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

true_positive = {key: 0 for key in candidate_labels}
false_positive = {key: 0 for key in candidate_labels}

for seq in sequences:
    total += 1
    label = output_map[seq]["label"]
    if seq in colors[label]:
        correct += 1
        true_positive[label] += 1
    else:
        false_positive[label] += 1


print(f"total: {total}, correct: {correct}, acc: {correct / total}")


recall = {}
precision = {}
for label in candidate_labels:
    recall[label] = true_positive[label] / len(colors[label])
    precision[label] = true_positive[label] / (true_positive[label] + false_positive[label])

print(f"recall:\n{json.dumps(recall, indent=2)}")
print(f"precision:\n{json.dumps(precision, indent=2)}")

f1_score = {}
for label in candidate_labels:
	# 避免除以零的错误
	if (precision[label] + recall[label]) > 0:
		f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
	else:
		f1_score[label] = 0

print(f"f1_score:\n{json.dumps(f1_score, indent=2)}")

import plotly.graph_objects as go
from plotly.offline import plot

# 假设你已经计算了这三个字典
# recall = {'Red': 0.62, 'Orange': 0.87, ...}
# precision = {'Red': 0.75, 'Orange': 0.89, ...}
# f1_score = {'Red': 0.68, 'Orange': 0.88, ...}

# 准备数据
labels = list(recall.keys())
recall_values = [recall[label] for label in labels]
precision_values = [precision[label] for label in labels]
f1_values = [f1_score[label] for label in labels]

# 创建图形对象
fig = go.Figure()

# 添加召回率柱状图
fig.add_trace(go.Bar(
	name='Recall',
	x=labels,
	y=recall_values,
	marker_color='blue'
))

# 添加精确率柱状图
fig.add_trace(go.Bar(
	name='Precision',
	x=labels,
	y=precision_values,
	marker_color='red'
))

# 添加 F1 分数柱状图
fig.add_trace(go.Bar(
	name='F1 Score',
	x=labels,
	y=f1_values,
	marker_color='green'
))

# 设置柱子为并排显示
fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.1)

# 更新布局和标题
fig.update_layout(
	title='Zero-Shot分类模型评估',
	xaxis_title='类别',
	yaxis_title='分数',
	yaxis_range=[0, 1]
)

# 输出到 HTML 文件
plot(fig, filename='zeroshot_estimate.html', auto_open=True)

print("柱状图已保存到 zeroshot_estimate.html 文件中，并已自动打开。")
