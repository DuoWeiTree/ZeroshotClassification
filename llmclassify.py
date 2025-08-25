import random
from typing import List, Any
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
import time
import os
import copy

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
		if n <= 0:
			return []
		if n > len(self.indices):
			raise ValueError("剩余元素不足以完成本次抽样。")
		
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

	def dumps(self):
		return json.dumps({
			"cnt": self.cnt,
			"ns": self.ns,
			"N_total": self.N_total
		})

	def loads(self, s: str):
		tmp = json.loads(s)
		self.cnt = tmp["cnt"]
		self.ns = tmp["ns"]
		self.N_total = tmp["N_total"]

class llmclassifier:
	class UserInput(BaseModel):
		existing_categories: List[str] = Field(..., description="已知的类别列表，可能为空")
		data_to_classify: List[str] = Field(..., description="待分类的文本列表，不会为空")
	class ClassificationOutput(BaseModel):
		all_categories: List[str] = Field(..., description="最终的类别列表，包含已有类别和新识别的类别，避免重复或语言差异造成的多余分类")
		classification_results: List[str] = Field(..., description="每个待分类文本所属的类别，严格引用all_categories中的类名")
		# merged_categories: List[Dict[str, str]] = Field(..., description="旧类别被合并到新类别的映射，记录语言统一或合并情况")

	def __init__(self, column_name: str):

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
- 处理 'data_to_classify' 列表中的每一项文本。
- 将每条文本映射到 'all_categories' 中最合适的类别。
- 如果某条文本与现有类别不匹配，你必须新建一个类别，并加入到 'all_categories'。
- 输出的类别名称必须严格来自 'all_categories'，不可自创或使用列表外的名称。
- 绝对不可以合并或删除已有类别。

输出格式必须严格遵循如下 JSON：
Output Schema:
{output_schema_str}
"""

		import llmagent
		self.agent = llmagent.LLMLite(system_prompt=system_prompt, model="gpt-4o")
		# agent = llmagent.Gemini(system_prompt=system_prompt, model="gemini-2.5-flash")

	def request(self, existing_categories: list[str], data_to_classify: list[str]):
		user_prompt = {
			"existing_categories": existing_categories,
			"data_to_classify": data_to_classify,
		}

		raw_output = self.agent.request(json.dumps(user_prompt, indent=2, ensure_ascii=False))
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

class llmmerger:
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

	def __init__(self, column_name: str):
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
		self.agent = llmagent.LLMLite(system_prompt=system_prompt, model="gpt-4o")
		# agent = llmagent.Gemini(system_prompt=system_prompt, model="gemini-2.5-flash")

	def request(self, categories: list[str], samples: list[str]):
		user_prompt = {
			"categories": categories,
			"samples": samples
		}

		raw_output = self.agent.request(json.dumps(user_prompt, indent=2, ensure_ascii=False))
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

def good_turing(tags: list[str], column_name: str, N=100, min_N_total = 1000, T: float=0.01):
	# 构建system_prompt

	sampler = Sampler(tags)

	N_total = 0
	r0 = 1
	counter = Counter()

	"all_categories"
	"classification_results"
	"merged_categories"

	classifier = llmclassifier(column_name=column_name)
	merger = llmmerger(column_name=column_name)


	while r0 > T or N_total < min_N_total:
		try:
			samples = sampler.get_samples(n=N)
			N_total += N
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

		if 1 in counter.ns:
			r0 = counter.ns[1] / N_total
			print(f"------------------->>> r0 = {r0} <<<-------------------")

		
	
	if r0 > T:
		print(f"样本数不足, r0={r0} > T={T}")
	else:
		print(f"分类完成, r0={r0} < T={T}, cnt:\n{json.dumps(counter.cnt,indent=4, ensure_ascii=False)}")

	existing_categories = [k for k, v in counter.cnt.items()]
	return existing_categories

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

	def extend(self, tags: list[str], column_name: str, N=100, min_N_total = 1000, T: float=0.01):
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

		classifier = llmclassifier(column_name=column_name)
		merger = llmmerger(column_name=column_name)

		while counter.r0() > T or counter.N_total < min_N_total:
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

			print(f"------------------->>> r0 = {counter.r0()} <<<-------------------")

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

		return self.get_categories(), final_merge
	
	@staticmethod
	def merge_counter(ct1: Counter, ct2: Counter, merger: llmmerger, samples) -> tuple[Counter, dict]:
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
