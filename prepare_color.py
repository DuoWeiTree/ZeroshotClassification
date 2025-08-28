import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet
# 查找'car'的所有同义词集

candidate_label = [
	'Red',
	'Orange',
	'Yellow',
	'Green',
	'Cyan',
	'Blue',
	'Purple',
	'Magenta',
	'Pink',
	'Brown',
	'Gray',
	'Black',
	'White'
]

for label in candidate_label:

	synonyms = wordnet.synsets(label)

	# 打印每个同义词集及其对应的近义词
	for syn in synonyms:
		print(f"同义词集: {syn.name()}")
		print(f"定义: {syn.definition()}")
		print("-----------------------")
		for lemma in syn.lemmas():
			print(f"	近义词: {lemma.name()}")