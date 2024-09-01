# -*- coding: utf-8 -*-

"""
call the apis of different llms

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import random
import json
import traceback
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from copy import deepcopy
import optparse

# import dashscope
# from http import HTTPStatus
# import openai

class LLM_api:
	def __init__(self, llm='tyqianwen', idx=None):
		self.llm = llm
		if llm == 'tyqianwen':
			self.api = TYQW_api('qwen-max-0403')   #('qwen-max-0107')

		elif llm == 'gpt3':
			self.api = ChatGPTKey()

		elif llm == 'baichuan':
			self.api = Baichuan_api()

		elif llm == 'llama':
			self.api = LLama_api('llama3_70b')
		else:
			print('unknown LLM')		
		

class TYQW_api:
	"""docstring for TYQW_api"""
	def __init__(self, model):	
		
		my_key = 'api_key*****' 
		dashscope.api_key = my_key
		self.model = model
		print('model:', self.model)

	def tokenizer(self, content):
		response = dashscope.Tokenization.call(model='qwen-max-1201',   #qwen-max
									 messages=[{'role': 'user', 'content': content}],
									 )
		if response.status_code == HTTPStatus.OK:
			print('Result is: %s' % response)
		else:
			print('Failed request_id: %s, status_code: %s, code: %s, message:%s' %
				  (response.request_id, response.status_code, response.code,
				   response.message))

	def call_with_messages(self, content):
		messages = [#{'role': 'system', 'content': 'You are a helpful assistant.'},
					{'role': 'user', 'content': content}]
		response = dashscope.Generation.call(
			model = self.model, #'qwen-max-1201',
			messages=messages,
			seed=1234, #random.randint(1, 10000),
			result_format='message',  # set the result to be "message" format.
		)
		if response.status_code == HTTPStatus.OK:
			pass 

		else:
			print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
				response.request_id, response.status_code,
				response.code, response.message
			))
			response = 'error calling'

		return response

	def get_response(self, content):
		try:
			response = self.call_with_messages(content)
			if isinstance(response, str): 
				time.sleep(1)
				response = self.call_with_messages(content) 

			if not isinstance(response, str):
				response = response.output['choices'][0]['message']['content']
			
		except:
			traceback.print_exc()
			return 'error calling'

		return response

class ChatGPTKey:
	"""docstring for ChatGPT"""
	def __init__(self, model='gpt-3.5-turbo'):
		api_key = 'api_key*****'  
		self.model = model
		self.template_data = {
			"model": model,
			"messages": [
				{
					"role": "user",
					"content": "hello",
				}
			],
			"temperature": 0,
		}
		print('current LLM is:', model)
		openai.api_key = api_key

	def get_response(self, content):
		try:
			print('create model')
			response = openai.ChatCompletion.create(
				model=self.model,
				messages=[{"role":"user", "content":content}],
				temperature=0,)
			
			res = response["choices"][0]["message"]["content"]
		except Exception as e:
			traceback.print_exc()
			print('failure during calling api')
			res = ''

		if len(res) == 0:
			time.sleep(1)
			try:
				print('create model for the second time')
				response = openai.ChatCompletion.create(
					model=self.model,
					messages=[{"role":"user", "content":content}],
					temperature=0,)
				
				res = response["choices"][0]["message"]["content"]
			except Exception as e:
				traceback.print_exc()
				print('failure during calling api')
				res = 'error calling'	

		return res

class Baichuan_api:
	"""docstring for  Baichuan_api"""
	def __init__(self, model="Baichuan3-Turbo"):
		self.url = "https://api.baichuan-ai.com/v1/chat/completions"
		self.api_key = "api_key*****"

		self.template_data = {
			"model": model, # "Baichuan2-Turbo",
			"messages": [{
				"role":"user",
				"content": ""
			}],
			"temperature": 0,
			"stream": False,
			# "max_tokens":10
		}
		print('current LLM is:', model)

		self.headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " +self.api_key
		}

	def request_api(self, data):
		post = ''
		try:
			post = requests.post(self.url, headers=self.headers, data=json.dumps(data))
			content = ''
			if post.status_code == 200:
				if self.template_data['stream']:
					for line in post.iter_lines():
						if line:
							content += line.decode('utf-8')
				else:
					content = json.loads(post.text)['choices'][0]['message']['content']
			else:
				print("Request failed, status code:", post.status_code)
				print("Request failed, X-BC-Request-Id:", post.headers.get("X-BC-Request-Id"))

			return content

		except Exception as e:
			traceback.print_exc()
			print('failure during calling api')
			return ''

	def get_response(self, content):
		self.template_data['messages'][0]['content'] = content
		response = self.request_api(self.template_data)
		if len(response) == 0: # 出错
			time.sleep(3)
			response = self.request_api(self.template_data)
		
		try:
			if len(response) == 0:
				response = 'error calling'
		except:
			traceback.print_exc()
			return 'error calling'

		return response


class LLama_api:
	def __init__(self, model="llama_2_13b"):

		print('current LLM is:', model)
		if '3_70b' in model:
			model_id = 'Meta-Llama-3-70B-Instruct'
		elif '3_8b' in model:
			model_id = 'Meta-Llama-3-8B-Instruct'
		elif '2_13b' in model:
			model_id = 'Llama-2-13b-chat-hf'

		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16, device_map='auto')

	def request_api(self, messages):
		inputs = self.tokenizer(messages, return_tensors="pt")
		outputs = self.model.generate(inputs.input_ids, do_sample=False, max_length=20+len(messages))
		final_result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
		res = final_result[0]
		return res

	def get_response(self, content):
		return self.request_api(content)


if __name__ == '__main__':
	# llm = 'tyqianwen'
	# llm = 'gpt3'
	# llm = 'gptk'
	llm = 'baichuan'

	mdopt = optparse.OptionParser()
	mdopt.add_option('-i', '--index', dest='index', type='int', default=0)
	options, args = mdopt.parse_args()
	idx = options.index

	llm_api = LLM_api(llm, idx)
	# llm_api.get_understandp_res()
	
	# llm_api.get_errorclean_res()

	# llm_api.get_translate_res()

	llm_api.get_cufencheck_res()


