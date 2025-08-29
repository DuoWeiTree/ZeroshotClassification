import random
from typing import List, Any
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
import time
import copy
from typing import Callable, Iterable
import pandas as pd
import numpy as np
from sortedcontainers import SortedSet
import pickle
from tqdm import tqdm

class Sampler:
	"""
	一个用于不重复抽样的类。抽样基于样本的位置，确保每个样本只被抽取一次。
	"""
	def __init__(self, population: List[Any]):
		"""
		初始化 Sampler。
		Args:
			population: 包含所有元素的总体列表。
		"""
		self.population = population
		# 存储所有样本的初始索引
		self.indices = list(range(len(population)))
		
	def get_samples(self, n: int) -> List[Any]:
		"""
		每次调用时返回一个不重复的、随机抽样的列表。
		Args:
			n: 每次抽样的数量。
		Returns:
			一个包含 n 个不重复元素的列表。
		Raises:
			ValueError: 如果剩余元素不足 n 个。
		"""
		if len(self.indices) == 0:
			raise ValueError("列表已空，无法继续采样")
		
		if n <= 0:
			return []
		if n >= len(self.indices):
			sampled_indices = self.indices.copy()
		else:
			# 从剩余索引中随机抽取 n 个不重复的索引
			sampled_indices = random.sample(self.indices, n)
		
		# 从剩余索引列表中移除已抽取的索引
		for index in sampled_indices:
			self.indices.remove(index)
		
		# 根据抽取的索引返回相应的样本
		samples = [self.population[i] for i in sampled_indices]
		
		return samples

	def reset(self):
		"""
		重置抽样器，恢复所有索引，可以重新开始抽样。
		"""
		self.indices = list(range(len(self.population)))

class Counter:
	def __init__(self):
		self.cnt = {}
		self.ns = {
			0: 0,
			1: 0
		}
		self.N_total = 0

	def ensure(self, category):
		if category not in self.cnt:
			self.cnt[category] = 0
			self.ns[0] += 1
			
	def add(self, category, number: int = 1):
		if category not in self.cnt:
			self.cnt[category] = 0
			self.ns[0] += 1
		
		self.ns[self.cnt[category]] -= 1
		self.cnt[category] += number
		self.N_total += number

		if self.cnt[category] not in self.ns:
			self.ns[self.cnt[category]] = 0
		self.ns[self.cnt[category]] += 1

	def merge(self, old, new):
		if old == new or old not in self.cnt:
			return
		if new not in self.cnt:
			self.cnt[new] = 0
			self.ns[0] += 1
		self.ns[self.cnt[old]] -= 1
		self.ns[self.cnt[new]] -= 1
		self.cnt[new] += self.cnt[old]
		self.cnt.pop(old)
		if self.cnt[new] not in self.ns:
			self.ns[self.cnt[new]] = 0
		self.ns[self.cnt[new]] += 1

	def r0(self):
		return self.ns[1] / self.N_total if self.N_total > 0 else 1

class LLMClassifier:
	class UserInput(BaseModel):
		existing_categories: List[str] = Field(..., description="已知的类别列表，可能为空")
		data_to_classify: Dict[int, str] = Field(..., description="待分类的文本字典，键为整数索引，值为文本，不会为空")

	class ClassificationOutput(BaseModel):
		all_categories: List[str] = Field(..., description="最终的类别列表，包含已有类别和新识别的类别，避免重复或语言差异造成的多余分类")
		classification_results: Dict[int, str] = Field(..., description="每个待分类文本所属的类别映射，键为输入的索引，值严格引用all_categories中的类名")
		# merged_categories: List[Dict[str, str]] = Field(..., description="旧类别被合并到新类别的映射，记录语言统一或合并情况")

	def __init__(self, column_name: str, model_name = "gpt-4o"):
		input_schema_str = json.dumps(self.UserInput.model_json_schema(), ensure_ascii=False, indent=2)
		output_schema_str = json.dumps(self.ClassificationOutput.model_json_schema(), ensure_ascii=False, indent=2)

		system_prompt = f"""
你是一个专业的文本分类助手。
从以下维度进行分类:
{column_name}

分类原则：
1. **唯一标准**：只能按照一个统一标准进行分类（参考 {column_name}）。
2. **中心语义**：类名应当代表类内文本的共同中心语义。
3. **优先归类**：尽可能将文本归入 'existing_categories' 中已有的类别。
4. **严格映射**：所有输出类别名称必须严格来自 'all_categories'。
   - 如果需要新类别，你必须先在 'all_categories' 中加入该类别，再将文本映射到它。
   - 不允许输出任何不在 'all_categories' 中的类别名称。
5. **禁止合并**：不允许合并、删除或吸收已有类别。
   - 每个已有类别必须保留，不能因为语义相似而去掉或合并。
6. **禁止上位类创建**：不允许生成仅为已有类别的直接上位概念或包容性更强的类别。
7. **语言统一**：不同语言表达的相同含义，应映射为同一英文类别名。
8. **保持具体性**：类别必须足够具体和细化，避免过度笼统。类别名要让用户一眼就能看出其代表的对象或含义。

用户输入模式（UserInput Schema）：
{input_schema_str}

你的任务：
- 处理 'data_to_classify' 字典中的每一项文本。
- 将每条文本映射到 'all_categories' 中最合适的类别。
- 如果某条文本与现有类别不匹配，你必须新建一个类别，并加入到 'all_categories'。
- 输出的类别名称必须严格来自 'all_categories'，不可自创或使用列表外的名称。
- 绝对不可以合并或删除已有类别。

输出格式必须严格遵循如下 JSON：
Output Schema:
{output_schema_str}
"""

		import llmagent
		self.agent = llmagent.LLMLite(system_prompt=system_prompt, model=model_name)

	def request(self, existing_categories: list[str], data_to_classify: dict[int, str], verbose=False):
		user_prompt = {
			"existing_categories": existing_categories,
			"data_to_classify": data_to_classify,
		}

		raw_output = self.agent.request(json.dumps(user_prompt, indent=2, ensure_ascii=False) ,verbose=verbose)
		return self.parse_llm_response(raw_output)

	def parse_llm_response(self, response_text: str) -> ClassificationOutput:
		"""
		解析并验证LLM返回的JSON字符串。
		这个函数会预处理响应，移除可能存在的Markdown代码块。

		Args:
			response_text: LLM返回的原始文本。

		Returns:
			一个 ClassificationOutput 实例。

		Raises:
			ValidationError: 如果JSON格式正确但内容不符合Pydantic模型。
			json.JSONDecodeError: 如果文本不是有效的JSON。
		"""
		# 1. 预处理：移除Markdown代码块
		cleaned_text = response_text.strip()
		if cleaned_text.startswith("```json"):
			# 移除开头的 ```json 和结尾的 ```
			cleaned_text = cleaned_text[7:].strip().rstrip("`")
		
		# 2. 解析和验证
		return self.ClassificationOutput.model_validate_json(cleaned_text)

class LLMMerger:
	class UserInput(BaseModel):
		categories: List[str] = Field(..., description="待合并的类别名列表，不会为空")
		samples: List[str] = Field(
			..., 
			description="原始未分类数据样本，仅用于参考层级关系，帮助判断类别归并的合适层级，不用于直接合并"
		)

	class MergeOutput(BaseModel):
		merged_categories: Dict[str, str] = Field(
			..., 
			description="key类被合并到value类，保证最终类名语义唯一、英文统一"
		)

	def __init__(self, column_name: str, model_name = "gpt-4o"):
		input_schema_str = json.dumps(self.UserInput.model_json_schema(), ensure_ascii=False, indent=2)
		output_schema_str = json.dumps(self.MergeOutput.model_json_schema(), ensure_ascii=False, indent=2)

		system_prompt = f"""
你是一个专业的商品类别**语义消歧助手**。
你的唯一任务是：接收一个类别名列表，并输出严格的消歧合并结果。

⚠️ 核心原则（必须遵守）：
1. **仅限概念包围合并**：只有当一个类别的语义被另一个类别 **完全覆盖** 时，才能合并。
   - 覆盖：即 A 的所有含义都包含在 B 内，或 B 的所有含义都包含在 A 内。
   - 若只是相似、关联、部分交集，**一律不得合并**。
2. **禁止过度合并**：绝对不能因为相似度高、同类联想、属于同一大类而合并。
   - 例：`Laptop` 和 `Tablet` 不能合并。
   - 例：`Shoes` 和 `Sneakers` 可以合并（因为 Sneakers 完全属于 Shoes）。
3. **禁止泛化**：不能因为找不到共同点就合并到更宽泛的大类。
4. **中心语义**：合并后类别名必须精准表达共同语义，不能模糊。
5. **保留差异**：只要类别之间有区分价值，必须保持独立。
6. **优先覆盖**：若存在包围关系，较小的类别并入能覆盖它的较大类别。
7. **新类生成**：无法归并的类别保持独立。
8. **语言统一**：不同语言表达相同含义时，合并为统一英文名。
9. **禁止自合并**：原始类别不能映射到自身。
10. **数量限制**：输出的合并映射最多 3 对。
11. **不可合并同词性但程度/规格不同的类别**：例如 `XXS` 与 `XS`，即使属于同一大类，也**绝对不可合并**。

你将收到一个JSON对象，其格式严格遵循如下的用户输入模式（UserInput Schema）。
UserInput Schema:
{input_schema_str}

你必须输出：
- 一个合并映射字典，键为原始类别，值为其对应的最终类别（不能映射到自身）。

最终输出必须严格符合以下JSON格式：
Output Schema:
{output_schema_str}
"""

		import llmagent
		self.agent = llmagent.LLMLite(system_prompt=system_prompt, model=model_name)
		# agent = llmagent.Gemini(system_prompt=system_prompt, model="gemini-2.5-flash")

	def request(self, categories: list[str], samples: list[str], verbose=False):
		user_prompt = {
			"categories": categories,
			"samples": samples
		}

		raw_output = self.agent.request(json.dumps(user_prompt, indent=2, ensure_ascii=False), verbose=verbose)
		return self.parse_llm_response(raw_output)

	def parse_llm_response(self, response_text: str) -> MergeOutput:
		"""
		解析并验证LLM返回的JSON字符串。
		这个函数会预处理响应，移除可能存在的Markdown代码块。

		Args:
			response_text: LLM返回的原始文本。

		Returns:
			一个 ClassificationOutput 实例。

		Raises:
			ValidationError: 如果JSON格式正确但内容不符合Pydantic模型。
			json.JSONDecodeError: 如果文本不是有效的JSON。
		"""
		# 1. 预处理：移除Markdown代码块
		cleaned_text = response_text.strip()
		if cleaned_text.startswith("```json"):
			# 移除开头的 ```json 和结尾的 ```
			cleaned_text = cleaned_text[7:].strip().rstrip("`")
		
		# 2. 解析和验证
		return self.MergeOutput.model_validate_json(cleaned_text)

class CategoryExtractor:
	def __init__(self, cache_path:str = None):
		"""
		初始化 CategoryExtractor 类。
		设置一个计数器对象并尝试从指定的缓存路径加载数据。

		Args:
			cache_path (str, optional): 缓存文件的路径。如果为 None，则不加载缓存。
		"""
		self.counter = Counter()
		self.cache_path = cache_path
		if self.cache_path is not None:
			print(f"尝试加载位于 {self.cache_path} 的缓存")
			self.load_from(self.cache_path)

	def get_categories(self):
		"""
		获取当前计数器中所有的类别。

		Returns:
			list[str]: 包含所有类别名称的列表。
		"""
		all_categories = [key for key, value in self.counter.cnt.items()]
		return all_categories

	def load_from(self, filename: str):
		"""
		从指定的文件中加载计数器状态。
		如果加载失败，将恢复到之前的状态。

		Args:
			filename (str): 要加载的文件的路径。
		"""
		try:
			with open(filename, "r", encoding="utf-8") as file:
				tmp = file.read()
				counter = Counter()
				counter.loads(tmp)
				self.counter = counter
			print(f"成功读取存档 {filename}")
		except Exception as e:
			print(f"读取存档失败 {e}")

	def save_to(self, filename: str):
		"""
		将当前计数器的状态保存到指定的文件中。
		如果保存失败，将打印错误信息。

		Args:
			filename (str): 要保存的文件的路径。
		"""
		try:
			with open(filename, "w", encoding="utf-8") as file:
				tmp = self.counter.dumps()
				file.write(tmp)
			print(f"成功保存至 {filename}")
		except Exception as e:
			print(f"保存失败 {e}")

	def extend(self, tags: list[str], column_name: str, N=100, min_N_total = 1000, T: float=0.02, max_iteration = 10, return_sampler = False):
		"""
		使用大语言模型对给定的标签进行分类和合并，以扩展现有类别。
		该方法会迭代地进行分类和合并，直到类别稳定或达到最小样本数。

		Args:
			tags (list[str]): 待处理的标签列表。
			column_name (str): 用于 LLM 模型的列名。
			N (int): 每次分类的样本数。默认为 100。
			min_N_total (int): 最小处理的样本总数。默认为 1000。
			T (float): 稳定性阈值。默认为 0.01。

		Returns:
			tuple[list[str], dict]: 一个元组，包含最终的类别列表和最终的合并映射字典。
		"""
		sampler = Sampler(tags)
		counter = Counter()

		"all_categories"
		"classification_results"
		"merged_categories"

		classifier = LLMClassifier(column_name=column_name)
		merger = LLMMerger(column_name=column_name)

		iteration = 0
		while (counter.r0() > T or counter.N_total < min_N_total) and iteration < max_iteration:
			try:
				samples = sampler.get_samples(n=N)
			except:
				break

			success = False
			while not success:
				try:
					existing_categories = [k for k, v in counter.cnt.items()]
					classify_output = classifier.request(existing_categories, samples)

					print("分类成功")
					print(f"All Categories:\n{classify_output.all_categories}\n")
					print(f"Classification results:\n{classify_output.classification_results}\n")
					print("------------------------------------------------------")

					for result in classify_output.classification_results:
						counter.add(result)
					success = True
				except ValidationError as e:
					print("JSON 验证失败！")
					print(e.json()) # 打印详细的错误信息，便于调试
				except json.JSONDecodeError as e:
					print("JSON 解析失败！")
					print(e)
				if not success:
					time.sleep(2)
			
			success = False
			while not success:
				try:
					all_categories = [k for k, v in counter.cnt.items()]
					merge_output = merger.request(all_categories, samples)

					print("合并成功")
					print(f"Merge Categories:\n{json.dumps(merge_output.merged_categories, indent=2, ensure_ascii=False)}\n")
					print("------------------------------------------------------")

					for old, new in merge_output.merged_categories.items():
						counter.merge(old, new)
					success = True
				except ValidationError as e:
					print("JSON 验证失败！")
					print(e.json()) # 打印详细的错误信息，便于调试
				except json.JSONDecodeError as e:
					print("JSON 解析失败！")
					print(e)
				if not success:
					time.sleep(2)
			iteration += 1
			print(f"------------------->>> Iteration {iteration}, r0 = {counter.r0()} <<<-------------------")

		if counter.r0() > T:
			print(f"样本数不足, r0 = {counter.r0()} > T = {T}")
		else:
			print(f"分类完成, r0 = {counter.r0()} < T = {T}, cnt:\n{json.dumps(counter.cnt,indent=4, ensure_ascii=False)}")


		existing_categories = [k for k, v in self.counter.cnt.items()]
		final_merge = {}
		if len(existing_categories):
			print("将新类别与已有类别进行合并")
			self.counter, final_merge = CategoryExtractor.merge_counter(self.counter, counter, merger, samples)
			print(f"合并后 r0 = {self.counter.r0()}")
		else:
			self.counter = counter


		if self.cache_path is not None:
			print(f"尝试保存缓存至 {self.cache_path}")
			self.save_to(self.cache_path)

		if return_sampler:
			return self.get_categories(), final_merge, sampler
		else:
			return self.get_categories(), final_merge
	
	@staticmethod
	def merge_counter(ct1: Counter, ct2: Counter, merger: LLMMerger, samples) -> tuple[Counter, dict]:
		"""
		将两个 Counter 对象进行合并，并使用 LLM 模型进行去重。

		Args:
			ct1 (Counter): 第一个计数器，新类别将合并到其中。
			ct2 (Counter): 第二个计数器，包含要合并的新类别。
			merger (llmmerger): 用于类别合并的 LLM 模型实例。
			samples: 用于 LLM 模型合并的样本数据。

		Returns:
			tuple[Counter, dict]: 一个元组，包含合并后的新计数器和合并映射字典。
		"""
		ct1 = copy.deepcopy(ct1)
		ct2 = copy.deepcopy(ct2)

		for tag, num in ct2.cnt.items():
			ct1.add(tag, num)

		success = False
		while not success:
			try:
				all_categories = [key for key, value in ct1.cnt.items()]
				merge_output = merger.request(all_categories, samples)

				print("合并成功")
				print(f"Merge Categories:\n{json.dumps(merge_output.merged_categories, indent=2, ensure_ascii=False)}\n")
				print("------------------------------------------------------")

				for old, new in merge_output.merged_categories.items():
					ct1.merge(old, new)
				success = True

			except ValidationError as e:
				print("JSON 验证失败！")
				print(e.json()) # 打印详细的错误信息，便于调试
			except json.JSONDecodeError as e:
				print("JSON 解析失败！")
				print(e)
			if not success:
				time.sleep(2)

		return ct1, merge_output.merged_categories

class Dictionary:
	def __init__(self):
		self.exist = SortedSet()
		self.set: dict[str, SortedSet] = {}

	def add(self, text: str, label: str):
		if text in self.exist:
			return

		if label not in self.set:
			self.set[label] = SortedSet()
		self.set[label].add(text)
		self.exist.add(text)

	def merge(self, from_label: str, to_label: str):
		if from_label not in self.set or from_label == to_label:
			return

		if to_label not in self.set:
			self.set[to_label] = SortedSet()
		
		self.set[to_label].update(self.set[from_label])
		self.set.pop(from_label)

	def reverse_map(self) -> dict[str, str]:
		return {text: label  for label, sorted_set in self.set.items() for text in sorted_set}

class LLMLabeler:
	"""LLMLabeler 用于批量文本分类和标签管理，支持缓存和 LLM 调用。"""

	def __init__(self, cache_path: str = None, model_name = "gpt-4o"):
		"""
        初始化 LLMLabeler。

        参数:
            cache_path (str, optional): 如果提供，初始化时会尝试加载缓存。
        """
		self.dic  = Dictionary()
		self.model_name = model_name
		self.cache_path = cache_path
		if cache_path is not None:
			print(f"尝试从 {cache_path} 加载缓存")
			self.load_from(cache_path)

	def load_from(self, file_path: str):
		"""
        从指定文件加载缓存。

        参数:
            file_path (str): pickle 文件路径。
        """
		try:
			with open(file_path, "rb") as file:
				self.dic = pickle.load(file)
			print(f"成功加载 {file_path}")
		except Exception as e:
			print(f"无法加载 {file_path} : {e}")

	def save_to(self, file_path: str):
		"""
        将当前分类缓存保存到指定文件。

        参数:
            file_path (str): pickle 文件路径。
        """
		try:
			with open(file_path, "wb") as file:
				pickle.dump(self.dic,file=file)
			print(f"成功保存至 {file_path}")
		except Exception as e:
			print(f"无法保存至 {file_path} : {e}")

	def extend(self, texts: list[str], column_name: str, batch_size: int = 512, verbose = False) -> None:
		"""
        批量分类 DataFrame 中指定列的文本，并尝试合并标签。

        参数:
            texts (list[str]): 待处理的文本列表。
            column_name (str): 要处理的列名 用于LLM提示词。
            batch_size (int, optional): 每次调用 LLM 的文本数量，默认 512。
            verbose (bool, optional): 是否打印详细日志，默认 False。

        功能:
            - 去重文本。
            - 排除已分类项。
            - 使用 LLM 对文本进行批量分类。
            - 自动处理 JSON 验证错误或解析错误，支持重试。
            - 尝试合并标签，减少类别冗余。
            - 更新内部 Dictionary。
            - 如果设置了 cache_path，则自动保存缓存。
        """
		unique_texts = np.unique(np.array(texts)).tolist()
		print(f"去重后剩余: {len(unique_texts)}")

		existing_map = self.dic.reverse_map()
		unique_texts = [t for t in unique_texts if t not in existing_map]
		print(f"去除已分类项后剩余: {len(unique_texts)}")

		sampler = Sampler(unique_texts)

		classifier = LLMClassifier(column_name, model_name=self.model_name)
		merger = LLMMerger(column_name, model_name=self.model_name)

		print(f"开始利用LLM分类")
		with tqdm(total=len(unique_texts)) as pbar:
			while True:
				try:
					samples = sampler.get_samples(batch_size)
				except:
					break
				success = False

				while not success:
					try:
						existing_categories = [k for k, v in self.dic.set.items()]

						data_to_classify = {idx: text for idx, text in enumerate(samples)}
						classify_output = classifier.request(existing_categories, data_to_classify, verbose=verbose)

						results = classify_output.classification_results
						tmp_map = {data_to_classify[key]: value for key, value in results.items() if key in data_to_classify}

						for text, label in tmp_map.items():
							self.dic.add(text, label)

						if verbose:
							print("------------------------------------------------------")
							print("分类成功")
							print(f"All Categories:\n{classify_output.all_categories}\n")
							print(f"Classification results:\n{tmp_map}\n")

						if len(results) != len(data_to_classify):
							if not verbose:
								print("------------------------------------------------------")
								print(f"Classification results:\n{tmp_map}\n")
							print(f"警告, 分类输出与输入不等长，请判断映射是否无误!")


						success = True
					except ValidationError as e:
						print("JSON 验证失败！")
						print(e.json()) # 打印详细的错误信息，便于调试
					except json.JSONDecodeError as e:
						print("JSON 解析失败！")
						print(e)
					if not success:
						time.sleep(2)
				
				success = False
				while not success:
					try:
						all_categories = [k for k, v in self.dic.set.items()]
						merge_output = merger.request(all_categories, samples, verbose=verbose)
						if verbose:
							print("------------------------------------------------------")
							print("合并成功")
							print(f"Merge Categories:\n{json.dumps(merge_output.merged_categories, indent=2, ensure_ascii=False)}\n")

						for from_label, to_label in merge_output.merged_categories.items():
							self.dic.merge(from_label, to_label)
						success = True
					except ValidationError as e:
						print("JSON 验证失败！")
						print(e.json()) # 打印详细的错误信息，便于调试
					except json.JSONDecodeError as e:
						print("JSON 解析失败！")
						print(e)
					if not success:
						time.sleep(2)
				
				pbar.update(len(samples))

		print(f"LLM分类完毕, 当前总类数 {len(self.dic.set)}")
		if self.cache_path:
			self.save_to(self.cache_path)

	def label(self, texts: list[str], column_name: str) -> list[str]:
		"""
        将 DataFrame 中的文本映射为已有分类标签，不调用 LLM。

        参数:
            texts (list[str]): 待处理的文本列表。
            column_name (str): 要处理的列名 用于LLM提示词。

        返回:
            list[str]: 文本对应的标签列表，未分类的文本返回空字符串 ""。
        """
		print(f"开始将 {len(texts)} 项映射至 {len(self.dic.set)} 类")
		all_map = self.dic.reverse_map()

		results = [all_map[t] if t in all_map else "" for t in texts]
		print("映射完毕")
		return results

	def extend_and_label(self, texts: list[str], column_name: str, batch_size: int = 512, verbose = False):
		"""
        先批量分类 DataFrame 中的文本，再返回对应标签列表。

        参数:
            texts (list[str]): 待处理的文本列表。
            column_name (str): 要处理的列名 用于LLM提示词。
            batch_size (int, optional): 每次调用 LLM 的文本数量，默认 512。
            verbose (bool, optional): 是否打印详细日志，默认 False。

        返回:
            list[str]: 文本对应的标签列表。
        """
		self.extend(texts, column_name, batch_size, verbose)
		return self.label(texts, column_name)

if __name__ == "__main__":

	import re
	from itertools import chain

	def spliter_0(text: str):
		return re.split("[,/&]+", text)

	def spliter_1(text: str):
		return text[2:-2].split("','")
	

	df = pd.read_csv("./tables/202508142031_产品信息.csv")
	column_name = "尺寸"
	texts = df[column_name].astype(str)

	# 分割
	texts_single = list(chain(*[spliter_0(text=t) for t in texts]))

	print(texts_single[:50])
	# 每个方法都有""""""文档可以看
	
	"""
	labeler.extend 增量更新
	labeler.label 映射标签
	labeler.extend_and_label 增量更新然后映射标签
	"""
	labeler = LLMLabeler(f"./cache/tmp-{column_name}", model_name="gpt-4o")
	# labeler = LLMLabeler(model_name="gemini-2.5-flash")
	label = labeler.extend_and_label(texts_single, column_name, batch_size=400)

	result = pd.DataFrame()
	result[column_name] = texts_single
	result["label"] = label

	result.to_csv(f"./labeled/tmp-{column_name}.csv")


	

	# ['REGULAR', 'XS', 'XS', 'XL', 'XS', 'XS', 'L', 'S', 'REGULAR', 'XL', 'XXL', 'XXL', 'S', 'S', 'L', 'M', 'REGULAR', 'M', 'S', 'XL', 'L', 'M', 'XL', 'M', 'L', 'XL', 'XS', 'XL', 'REGULAR', 'XL', 'L', 'S', 'S', 'REGULAR', 'REGULAR', 'S', 'REGULAR', 'XS', 'XL', 'REGULAR', 'S', 'XS', 'XS', 'XXL', 'XS', 'XL', 'S', 'XL', 'M', 'S', 'XS', 'REGULAR', 'XS', 'L', 'L', 'L', 'XL', 'L', 'M', 'XL', 'XL', 'XS', 'XL', 'SIZE_REFERENCE', 'XL', 'XL', 'L', 'XL', 'XL', 'REGULAR', 'L', 'L', 'M', 'XXS', 'XXXS', 'XXXS', 'SIZE_REFERENCE', 'L', 'M', 'S', 'L', 'S', 'M', 'XS', 'XS', 'XS', 'REGULAR', 'XL', 'M']