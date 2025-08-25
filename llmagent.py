import requests
import json

from google import genai
from google.genai import types

from typing import Iterable

class LLMLite:
	def __init__(self, system_prompt, model = "qwen-plus"):
		self.url = "http://192.168.66.206:4000/v1/chat/completions"

		self.headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer sk-PuGAvHoWvtZ8rbJdwL5PzQ"
		}
		self.system_prompt = system_prompt
		self.model = model
		
		self.messages = [{
			"role": "system",
			"content": self.system_prompt,
		}]

	def request(self, user_prompt) -> str:
		print("LLMLite Requesting...")
		messages = self.messages + [{
			"role": "user",
			"content": user_prompt,
		}]
		payload = {
			"model": self.model,
			"messages": messages,
		}
		# print(f"messages: {messages}")
		try:
			response = requests.post(self.url, headers=self.headers, data=json.dumps(payload, ensure_ascii=False))
			response.raise_for_status()
			response_msg = response.json()["choices"][0]["message"]["content"]
			print("LLMLite Request Success!")
			return response_msg
		
		except requests.exceptions.RequestException as e:
			print(f"请求失败: {e}")
		except Exception as e:
			print(f"未知错误: {e}")
		return None


class Gemini:
	def __init__(self, system_prompt: str = "", model: str = "gemini-2.5-flash-lite"):
		self.system_prompt = system_prompt
		self.model = model
		self.client = genai.Client(api_key="AIzaSyBfaQSHyyd6nGRnOHHq7zkI_9tTEbBo_vo")

	
	def request(self, tags):
		tags = list(tags)

		contents = f"{self.system_prompt}\n{tags}"

		with open("cache", "w", encoding="utf-8") as f:
			f.write(contents)

		response = self.client.models.generate_content(
			model=self.model,
			contents=contents,
			config=types.GenerateContentConfig(
				thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
			),
		)
		return response.text
	
		

