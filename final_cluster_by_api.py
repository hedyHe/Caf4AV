#coding:utf8
#directly cluster values based on answers from llm

import json
import os
import traceback
import re
import csv
import sys
from collections import Counter, defaultdict
import optparse
import requests
import random
import time
import copy

random.seed(1234)


from llm_api import TYQW_api
from llm_api import ChatGPTKey
from llm_api import LLama_api

class CluterbyApi(object):
	"""docstring for CluterbyApi"""
	def __init__(self, idx=-1, newprompt=9, break_thre=-1):
		self.dataset = 'dbpedia'
		self.dataset = 'cndbpedia'

		self.newprompt = newprompt     #different prompts template
		self.break_thre = break_thre  #threshold for early termination， -1 means the process will not end early
		self.break_thre4call = 200  #the max calls for LLM for each attributes
		self.jingjian = '0'	
		
		path, _ = os.path.split(os.path.realpath(__file__))
		self.ci_flag = '0'  
		
		self.cleanres_file = '/data/' + self.dataset + '_groundtruth.json'
		if self.dataset == 'dbpedia':
			self.newprompt = 3  
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f2_0.json')
			self.target_attrcons = ['timeZone', 'architectureType', 'chairLabel', 
									'stat1Header', 'batting', 'areaBlank1Title', 
									'chrtitle', 'broadcastArea', 
									'sworntype', 'lakeType', 'scoreboard', 'timezone1Dst', 
									'link2Name']   #13个 'thirdRiderMoto2Country', 'membersLabel', 


		else:   #cndbpedia
			self.newprompt = 10   
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f6_0.json')
			self.target_attrcons = ['技术性质_车站', '国际濒危等级_学科', '设备种类_科技产品', '形象特征_人物', 
									'大坝类型_水电站', '成因类型_地貌', '所处时间_学科', '民居类型_地点', 
									'交建筑类型_建筑', '剧目类型_戏剧', '性别_人物', '行业类型_机构', '节目属性_娱乐', 
									'展馆类型_机构', '文章状态_小说']	#15
									

		"""different prompts for chinese dataset
		0. 请问下面两个值算书籍的同一种类型吗？请回答是或不是。\n韩语\n外语
		1. 你是一名知识库质量管理者，正在对CNDBpedia这个知识库中的属性值进行聚类，对于小说状态这个属性下可能的取值有连载中、已完结、暂停更新、断更、TBC、预告/筹备中、草稿/试读版、修订中、废止/腰折、永久断更、定期更新、不定期更新、会员专享/付费章节、独家发布等，请问下面两个值算同一种小说状态吗？请回答是或不是。\n连载\n连载中
		2. 你是CNDBpedia知识库的质量管理者，正在对知识库中的属性值进行聚类，即给定某个属性下的两个取值，判断其是否是近义词，输出结果为“是”或“不是”。\n如：输入：属性“小说状态”下这两个取值是近义词吗？\n已完结\n完本\n输出：是\n输入：属性“小说状态”下这两个取值是近义词吗？\n未完结\n完本\n输出：不是\n输入：属性“小说状态”下这两个取值是近义词吗？\n更新中\n每天更新\n输出：是\n请问属性“小说状态”下这两个取值是近义词吗？
		"""
		if self.newprompt == 0:
			self.prompt_file = os.path.join(path, self.dataset, 'prompt0.json')
			self.read_prompts()
		elif self.newprompt == 1:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt1.json')
			self.read_prompts()
		elif self.newprompt == 2:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt2.json')
			self.read_prompts() 
		elif self.newprompt == 3:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt3.json')
			self.read_prompts() 
		elif self.newprompt == 4:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt4.json')
			self.read_prompts() 
		elif self.newprompt == 5:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt5.json')
			self.read_prompts() 
		elif self.newprompt == 6:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt6.json')
			self.read_prompts()
		elif self.newprompt == 7:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt7.json')
			self.read_prompts() 
		elif self.newprompt == 8:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt8.json')
			self.read_prompts() 
		elif self.newprompt == 9:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt9.json')
			self.read_prompts() 
		elif self.newprompt == 10:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt10.json')
			self.read_prompts()
		elif self.newprompt == 11:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt11.json')
			self.read_prompts()
		elif self.newprompt == -1:
			self.prompt_file = os.path.join(path, self.dataset, 'prompt.json')
			self.read_prompts() 
		else:
			print('unknow prompt templates', self.prompt)
			exit()

		print('target_attrcons', self.target_attrcons)

		self.llm = 'tyqianwen'
		self.llm = 'chatgpt'
		# self.llm = 'llama'
		# self.llm = 'llama2'

		if self.llm == 'tyqianwen':
			model = 'qwen-max-0403'
			self.llm_api = TYQW_api(model)
			prefix = 'qwen_max'
		elif self.llm == 'chatgpt':
			model = 'gpt-3.5-turbo'
			self.llm_api = ChatGPT(model)
			prefix = 'gpt35'
		elif self.llm == 'llama':
			model = 'llama_3_8b'
			prefix = 'llama3'
			self.llm_api = LLama_api(model)
		else:
			model = 'llama_2_13b'
			prefix = 'llama2'

		if self.newprompt >= 0:
			idx = 1
			self.llm_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_all_q_ans_newp'+str(self.newprompt)+'_'+str(idx)+'.json')
			self.llm_add_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_added_q_res_newp'+str(self.newprompt)+'_'+str(idx)+'.json')
			self.temp_llm_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_all_q_ans_newp'+str(self.newprompt)+'_'+str(idx+1)+'.json')
			
			if 'large_cp_w_f6' in self.sim_res_file:
				tag = 'large_cp_w_f6'
			elif 'large_cp_w_f2' in self.sim_res_file:
				tag = 'large_cp_w_f2'
			elif 'tfidf' in self.sim_res_file:
				tag = 'tfidf'
			elif 'large' in self.sim_res_file:
				tag = 'large'

			if self.break_thre == -1:	
				self.name = self.break_thre4call
			else:
				self.name = self.break_thre
			self.final_res_file = './final/'+self.dataset+'_'+prefix+'_res_'+tag+'_'+self.ci_flag+'_'+str(self.name)+'_newp'+str(self.newprompt)+'_'+self.jingjian+'_test.json'
			
			self.temp_final_res_file = './final/'+prefix+'/temp/'+self.dataset+'_temp_'+prefix+'_res_'+tag+'_'+self.ci_flag+'_'+str(self.name)+'_newp'+str(self.newprompt)+'_'+self.jingjian+'.json'
		else:
			idx = 9
			if self.dataset == 'cndbpedia':
				tag1 = '03m_'
				if 'large_cp_w' in self.sim_res_file:
					tag = 'large_cp_w_f6'
				elif 'tfidf' in self.sim_res_file:
					tag = 'tfidf'
				elif 'large' in self.sim_res_file:
					tag = 'large'

			elif self.dataset == 'dbpedia':
				tag1 = '02m_'
				if 'large_cp_w' in self.sim_res_file:
					tag = 'large_cp_w_f2'
				elif 'tfidf' in self.sim_res_file:
					tag = 'tfidf'
				elif 'large' in self.sim_res_file:
					tag = 'large'

			self.llm_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_all_questions_ans_'+tag1+str(idx)+'.json')
			self.llm_add_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_added_q_res_'+tag1+str(idx)+'.json')
			self.temp_llm_res_file = os.path.join('./final', prefix, self.dataset+'_'+prefix+'_all_questions_ans_'+tag1+str(idx+1)+'.json')

			if self.break_thre == -1:	
				self.name = self.break_thre4call
			else:
				self.name = self.break_thre
			self.final_res_file = './final/'+self.dataset+'_'+prefix+'_res_'+tag+'_'+self.ci_flag+'_'+str(self.name)+'_'+self.jingjian+'_test.json'
			self.temp_final_res_file = './final/'+prefix+'/temp/'+self.dataset+'_temp_'+prefix+'_res_'+tag+'_'+self.ci_flag+'_'+str(self.name)+'_'+self.jingjian+'.json'
				
		
		if os.path.exists(self.temp_llm_res_file):
			print('The file already exists. You need to reset the idx', self.temp_llm_res_file)
			exit()
		if os.path.exists(self.llm_add_res_file):
			print('The file already exists. You need to reset the idx', self.llm_add_res_file)
			exit()

		print('dataset:', self.dataset)
		print('model:', self.llm, model)
		print('jingjian:', self.jingjian)
		print('break_thre:', self.break_thre)
		print('newprompt:', self.newprompt)
		print('sim_res_file:', self.sim_res_file)
		print('ci_flag:', self.ci_flag)
		print('llm_res_file:', self.llm_res_file)
		print('temp_llm_res_file:', self.temp_llm_res_file)
		print('final_res_file:', self.final_res_file)
		print('llm_add_res_file:', self.llm_add_res_file)
		print('temp_final_res_file:', self.temp_final_res_file)

	def read_prompts(self):
		with open(self.prompt_file, 'r', encoding='utf-8') as fr:
			self.new_prompts = json.load(fr)	

	def get_data(self):
		with open(self.cleanres_file, 'r', encoding='utf-8') as fr:
			temp_data = json.load(fr)
		
		data = {}
		keys = list(temp_data.keys())
		if self.dataset == 'cndbpedia':
			for k in keys:
				data[k] = {}
				for v in list(temp_data[k].keys()):
					if k+'_'+v in self.target_attrcons:
						data[k][v] = set()
						for c, vs in temp_data[k][v]['clusters'].items():
							data[k][v] |= set(vs)
						data[k][v] |= set(temp_data[k][v]['single_vs'])
						data[k][v] = list(data[k][v])

		elif self.dataset == 'dbpedia':
			for k in keys:
				if k in self.target_attrcons:
					v = 'all'
					data[k][v] = set()
					for c, vs in temp_data[k][v]['clusters'].items():
						data[k][v] |= set(vs)
					data[k][v] |= set(temp_data[k][v]['single_vs'])
					data[k][v] = list(data[k][v])

		return data

	def read_sims(self):
		# print('sim_res_file:', self.sim_res_file)
		all_sim_pairs = {}
		with open(self.sim_res_file, 'r', encoding='utf-8') as fr:
			line = fr.readline()
			while line:
				if 'the attribute currently processed is' in line:
					attr = fr.readline().strip()
					line = fr.readline()
					con = fr.readline().strip()
					line = fr.readline()
					if self.dataset == 'dbpedia':
						if attr not in self.target_attrcons:
							continue
					elif self.dataset == 'cndbpedia':
						if attr+'_'+con not in self.target_attrcons:
							continue

					if attr not in all_sim_pairs:
						all_sim_pairs[attr] = {}
					all_sim_pairs[attr][con] = {}

					# print(attr)
					try:
						sim_pairs = json.loads(line)
					except Exception as e:
						print(line)
						sim_pairs = {}
						traceback.print_exc()
						# exit()

					for k, sim in sim_pairs.items():
						(x, y) = k.split('--##--')
						all_sim_pairs[attr][con][(x,y)] = sim

				line = fr.readline()
				# break
		# print(all_sim_pairs.keys())
		return all_sim_pairs
	
	def read_llm_res(self):
		if os.path.exists(self.llm_res_file):
			print('the results of a LLM', self.llm_res_file)
			with open(self.llm_res_file, 'r', encoding='utf8') as fr:
				llm_res = json.load(fr)
		
		else:
			print('the file does not exist', self.llm_res_file)
			llm_res = {}
			if self.dataset == 'cndbpedia':
				for k in self.target_attrcons:
					attr, con = k.split('_')
					if attr not in llm_res:
						llm_res[attr] = {}
					llm_res[attr][con] = {}
			elif self.dataset == 'dbpedia':
				for k in self.target_attrcons:
					llm_res[k] = {'all':{}}
		return llm_res

	def get_llm_res(self, ans, llama=False, v0=None, v1=None, q=None):
		if not llama:
			ans = ans.lower()
			ans = ans.split('output')[-1]   #english
			ans = ans.split('输出')[-1]     #chinese
			if ans.startswith('：') or ans.startswith(':'):
				ans = ans[1:].strip()

			if re.search(r'(不是)|(不同)|(否)', ans):
				res = 'no'
			elif ans.startswith('是'):
				res = 'yes'
			elif ans.startswith('yes'):
				res = 'yes'
			elif ("算同一" in ans):
				res = 'yes'
			elif re.search(r'output.*(n/a)', ans):
				res = 'unsure' 
			elif ('无法确定' in ans) or ('不能确定' in ans):
				res = 'unsure'
			elif re.search(r'yes.*no', ans) or re.search(r'no.*yes', ans):
				res = 'unsure'
			elif ans.startswith('no'):
				res = 'no'
			elif ('不相同' in ans) or ('不完全相同' in ans) or ('不表示同一' in ans) or ('不完全属于' in ans):
				res = 'no'	
			elif re.search(r'output.*(no)', ans):
				res = 'no' 	
			elif re.search(r'answer.*(no)', ans):
				res = 'no' 	
			elif re.search(r'say.*(no)', ans):
				res = 'no' 	
			elif re.search(r'return.*(no)', ans):
				res = 'no' 	
			elif re.search(r'answer.*(yes)', ans):
				res = 'yes'
			elif re.search(r'output.*(yes)', ans):
				res = 'yes' 
			elif re.search(r'say.*(yes)', ans):
				res = 'yes'
			elif re.search(r'输出.*(是)', ans):
				res = 'yes'
			elif re.search(r'[(so)(therefore)].*(yes)', ans):
				res = 'yes'
			elif (' difficult to' in ans) or (' difficult to' in ans) or ('sorry' in ans) :
				res = 'unsure'
			elif (' hard to ' in ans)  or ('more detail' in ans) or ('more informa' in ans) or ('cannot' in ans) or ('can\'t' in ans):
				res = 'unsure'
			elif ('抱歉' in ans) or  ('无法' in ans):
				res = 'unsure'
			elif ans == '接口出错' or '出错' in ans:
				res = 'failure'
			else:
				res = 'unknow'	
		else:
			ans = ans.split('Input')
			target_res = []
			for y in ans:
				if '\n'+v0+'\n' in y and '\n'+v1+'\n' in y:
					target_res.append(y)
					
			if not target_res:
				res = 'unknow'

			else:      #If there are multiple results, just take the first result
				ans1 = target_res[0].split('Output')[-1].strip(': \n：')
				ans1 = ans1.split('\n')[0]
				if ans1 in ['yes']:
					res = 'yes'
				elif ans1 in ['no']:
					res = 'no'
				else:
					res = 'unsure'
					# print('*******************\nUnconsidered cases', i)
					# print('v0: ', v0)
					# print('v1:', v1) 
					# print('ans1:', ans1)
					
		return res

	def post(self, temp_res, all_data):
		#Post-processing clustering results, and considering transitive	
		#temp_res: similar pairs
		#all_data: all values need to be clustered
		mapping = {}  
		trans = {}
		res = {}
		temp_idx = 0

		for x in all_data.keys():
			new_x = x.split('#****#')[0]
			trans[new_x] = x

		for i in range(len(temp_res)-1, -1, -1):
			pair = temp_res[i]
			idx0 = mapping.get(pair[0], -1)
			idx1 = mapping.get(pair[1], -1)
			if idx0 != -1:
				if idx1 != -1:
					for v in res[idx1]:
						res[idx0].add(v)
						mapping[v] = idx0
				else:
					mapping[pair[1]] = idx0
					res[idx0].add(trans[pair[1]])
			elif idx0 == -1:
				if idx1 != -1:
					mapping[pair[0]] = idx1
					res[idx1].add(trans[pair[0]])
				else:
					idx = temp_idx
					res[idx] = set([trans[pair[0]], trans[pair[1]]])
					mapping[pair[0]] = idx
					mapping[pair[1]] = idx
					temp_idx += 1

		# print('all_data:', all_data)
		# print('res:', res)
		# print('mapping:', mapping)
		final_res = []
		for k, vs in res.items():
			final_res.append(list(vs))

		for x in all_data.keys():
			new_x = x.split('#****#')[0]
			if new_x not in mapping:
				final_res.append([x])	

		return final_res	

	def main_all_pair_jingjian(self):
		"""All candidate pairs are judged once.
		if a is similar with b and a is similar with c, then a is similar with c.
		"""
		data = self.get_data()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #total calling number
		api_temp = 0   #new added calling number
		que_chars = 0   #the number of chars for new queries
		res_chars = 0   #the number of chars for new answers

		for attr, info in data.items():
			if api_temp > 5000:
				break
			print('------------------\nthe attribute currently processed is: ', attr)
			final_res[attr] = {}
			mid_res[attr] = {}
			for con, c_info in info.items():
				if api_temp > 5000:
					break
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					temp_final_res = defaultdict(list)
					print('-----------the concept currently processed is: ', con)
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					mappings = defaultdict(list)   
					all_vs = list(c_info.keys())
					for i in range(len(all_vs)-1):
						v0 = all_vs[i].split('#****#')[0]
						added_idx = len(temp_final_res)   
						for j in range(i+1, len(all_vs)):
							v1 = all_vs[j].split('#****#')[0]
							#first judging wether the value has be added in one cluster
							idxs0 = mappings.get(v0, [])
							idxs1 = mappings.get(v1, [])
							if set(idxs0) & set(idxs1):
								continue

							api_total += 1
							if self.dataset == 'cndbpedia':
								p = v0 +'#****#'+ v1
								p1 = v1 +'#****#'+ v0
							elif self.dataset == 'dbpedia':
								p = v0.lower() +'#****#'+ v1.lower()	
								p1 = v1.lower() +'#****#'+ v0.lower()

							if p in temp_llm_res:
								ans = temp_llm_res[p] 
								if ans in ['no', 'unsure', 'failure', 'unknow']:
									continue
							else:
								if v1.lower() == v0.lower():
									ans = 'yes'	
								elif p1 in temp_llm_res:
									ans = temp_llm_res[p1]
									# print('p已有结果', p, ans)
									if ans in ['no', 'unsure', 'unknow']:   #the value pairs is not same
										continue
								else:
									q = prompt+v0+'\n'+v1
									que_chars += len(q)
									temp = [attr, q, con, v0, v1]

									api_temp += 1

									res = self.llm_api.get_response(q)
									res_chars += len(res)
									ans = self.get_llm_res(res)

									temp.insert(2, ans)   # answer after processing
									temp.append(res)	 #answer generated by LLMs
									
									if ans in ['failure']:
										continue

									added_llm_res.append(temp)
									temp_llm_res[p] = ans 	
									# break
									if ans in ['no', 'unsure', 'unknow']:
										continue

							mappings[v0].append(added_idx)
							mappings[v1].append(added_idx)
							temp_final_res[added_idx].append(all_vs[i])
							temp_final_res[added_idx].append(all_vs[j])
						# 	break
						# break
					llm_res[attr][con] = temp_llm_res
					mid_res[attr][con] = temp_final_res
					
				except:
					traceback.print_exc()

				try:
					# trans = {}
					# for x in c_info.keys():
					# 	new_x = x.split('#****#')[0]
					# 	trans[new_x] = x

					added_vs = set()
					for k, vs in temp_final_res.items():
						# temp_added_vs = set()
						# for x in vs:
						# 	temp_added_vs.add(trans[x])
						# added_vs |= temp_added_vs
						# final_res[attr][con].append(list(temp_added_vs))
						added_vs |= set(vs)
						final_res[attr][con].append(list(set(vs)))
						
					for x in all_vs:
						if x in added_vs:
							continue
						else:
							final_res[attr][con].append([x])
				
				except:
					traceback.print_exc()

				
			# break
		# print("added_llm_res:", added_llm_res)
		print('total calling number', api_total, 'new added calling number', api_temp)
		if added_llm_res:
			print('added_llm_res written into a file', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	

			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)				

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)		

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)


		print('the number of chars for new queries:', que_chars)
		print('the number of chars for new answers:', res_chars)
	
	def main_stopbythre_jingjian(self):
		#The threshold is used to end early the process and consider the transitivity between

		data = self.get_data()
		all_sim_pairs = self.read_sims()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #total calling number
		api_temp = 0   #new added calling number
		que_chars = 0   #the number of chars for new queries
		res_chars = 0   #the number of chars for new answers

		for attr, info in data.items():
			if api_temp > 8000:
				break
			print('------------------\nthe attribute currently processed is: ', attr)
			final_res[attr] = {}
			mid_res[attr] = {}
			for con, c_info in info.items():
				if api_temp > 8000:
					break
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					print('-----------the concept currently processed is: ', con)
					sims = all_sim_pairs[attr][con]
					sims = sorted(sims.items(), key=lambda x:x[1], reverse=True)
					mappings = defaultdict(list)   #cluster ids for each value
					all_vs = list(c_info.keys())
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					temp_final_res = defaultdict(list)
					no_count = 0   #the number fo consecutive no
					temp_apicount = 0   #the number of calls to LLMs under the current attribute
					for (pair, v) in sims:
						v0 = pair[0].split('#****#')[0]
						v1 = pair[1].split('#****#')[0]
						added_idx = len(temp_final_res)   
						idxs0 = mappings.get(v0, [])
						idxs1 = mappings.get(v1, [])
						if set(idxs0) & set(idxs1):
							continue	

						api_total += 1
						if self.dataset == 'cndbpedia':
							p = v0 +'#****#'+ v1
							p1 = v1 +'#****#'+ v0
						elif self.dataset == 'dbpedia':
							p = v0.lower() +'#****#'+ v1.lower()	
							p1 = v1.lower() +'#****#'+ v0.lower()

						if p in temp_llm_res:
							ans = temp_llm_res[p] 
						else:
							if p1 in temp_llm_res:
								ans = temp_llm_res[p1] 	
							elif v1.lower() == v0.lower():
								ans = 'yes'							
							else:
								# ans = 'unknow'
								if api_temp > 8000:
									break
								if temp_apicount > 4000:
									break
								q = prompt+v0+'\n'+v1 +'\n' #+"输出："
								if 'Input:' in q:
									q += 'Output:'

								que_chars += len(q)
								# print(q)
								temp = [attr, q, con, v0, v1]
								api_temp += 1
								temp_apicount += 1
								# i = random.randint(0,1)
								# if i == 0:
								# 	ans = 'yes'
								# 	res = 'yes'
								# else:
								# 	ans = 'no'
								# 	res = 'no'

								# continue
								res = self.llm_api.get_response(q)
								res_chars += len(res)
								ans = self.get_llm_res(res)
								
								if ans not in ['failure']:
									temp.insert(2, ans)   # answer after processing
									temp.append(res)	 #answer generated by LLMs
									added_llm_res.append(temp)
									temp_llm_res[p] = ans

						if ans in ['yes']:
							no_count = 0
							mappings[v0].append(added_idx)
							mappings[v1].append(added_idx)
							temp_final_res[added_idx].append(pair[0])
							temp_final_res[added_idx].append(pair[1])
						elif ans in ['no']:
							no_count += 1
							if no_count == self.break_thre:
								print('the process stop early', pair, v, ans)
								break
						
					if temp_llm_res:
						llm_res[attr][con] = temp_llm_res

					mid_res[attr][con] = temp_final_res
				except:
					traceback.print_exc()
	
				try:
					trans = {}
					for x in c_info.keys():
						new_x = x.split('#****#')[0]
						trans[new_x] = x

					added_vs = set()
					for k, vs in temp_final_res.items():
						temp_added_vs = set()
						for x in vs:
							temp_added_vs.add(trans[x])
						added_vs |= temp_added_vs
						final_res[attr][con].append(list(temp_added_vs))
						
					for x in all_vs:
						if x in added_vs:
							continue
						else:
							final_res[attr][con].append([x])
				except:
					traceback.print_exc()

			# 	break
			# break
		# print("added_llm_res:", added_llm_res)
		print('total calling number', api_total, 'new added calling number', api_temp)
		if added_llm_res:
			print('added_llm_res written into a file', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	
			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)					

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)	

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)

		print('the number of chars for new queries:', que_chars)
		print('the number of chars for new answers:', res_chars)

	def main_stopbythre2_jingjian(self):
		#The threshold is used to end early the process
		data = self.get_data()
		all_sim_pairs = self.read_sims()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #total calling number
		api_temp = 0   #new added calling number
		que_chars = 0   #the number of chars for new queries
		res_chars = 0   #the number of chars for new answers

		for attr, info in data.items():
			print('------------------\nthe attribute currently processed is: ', attr)
			final_res[attr] = {}
			mid_res[attr] = {}
			for con, c_info in info.items():
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					print('-----------the concept currently processed is: ', con)
					sims = all_sim_pairs[attr][con]
					sims = sorted(sims.items(), key=lambda x:x[1], reverse=True)
					mappings = defaultdict(list)   
					all_vs = list(c_info.keys())
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					temp_final_res = defaultdict(list)
					no_count = 0   #the number of consecutive queries whose answer is "no"
					temp_apicount = 0   #the number of calling under the current attributes
					for (pair, v) in sims[:self.break_thre4call]:   #Only the first break_thre4call are considered
						v0 = pair[0].split('#****#')[0]
						v1 = pair[1].split('#****#')[0]
						added_idx = len(temp_final_res)   
						idxs0 = mappings.get(v0, [])
						idxs1 = mappings.get(v1, [])
						if set(idxs0) & set(idxs1):
							continue	

						api_total += 1
						if self.dataset == 'cndbpedia':
							p = v0 +'#****#'+ v1
							p1 = v1 +'#****#'+ v0
						elif self.dataset == 'dbpedia':
							p = v0.lower() +'#****#'+ v1.lower()	
							p1 = v1.lower() +'#****#'+ v0.lower()

						if p in temp_llm_res:
							ans = temp_llm_res[p] 
						else:
							if p1 in temp_llm_res:
								ans = temp_llm_res[p1] 	
							elif v1.lower() == v0.lower():
								ans = 'yes'							
							else:
								q = prompt+v0+'\n'+v1 +'\n' #+"输出："
								que_chars += len(q)
								# print(q)
								temp = [attr, q, con, v0, v1]
								api_temp += 1
								temp_apicount += 1

								res = self.llm_api.get_response(q)
								res_chars += len(res)
								ans = self.get_llm_res(res)
								
								if ans not in ['failure']:
									temp.insert(2, ans)   # answer after processing
									temp.append(res)	 #answer before processing
									added_llm_res.append(temp)
									temp_llm_res[p] = ans

						if ans in ['yes']:
							no_count = 0
							mappings[v0].append(added_idx)
							mappings[v1].append(added_idx)
							temp_final_res[added_idx].append(pair[0])
							temp_final_res[added_idx].append(pair[1])
						elif ans in ['no']:
							no_count += 1
							if no_count == self.break_thre:
								print('end early', pair, v, ans)
								break
						
					if temp_llm_res:
						llm_res[attr][con] = temp_llm_res

					mid_res[attr][con] = temp_final_res
				except:
					traceback.print_exc()
	
				try:
					trans = {}
					for x in c_info.keys():
						new_x = x.split('#****#')[0]
						trans[new_x] = x

					added_vs = set()
					for k, vs in temp_final_res.items():
						temp_added_vs = set()
						for x in vs:
							temp_added_vs.add(trans[x])
						added_vs |= temp_added_vs
						final_res[attr][con].append(list(temp_added_vs))
						
					for x in all_vs:
						if x in added_vs:
							continue
						else:
							final_res[attr][con].append([x])
				except:
					traceback.print_exc()

			# 	break
			# break
		# print("added_llm_res:", added_llm_res)
		print('total calling number', api_total, 'new added calling number', api_temp)
		if added_llm_res:
			print('added_llm_res written into a file', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	
			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)					

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)	

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)

		print('the number of chars for new queries:', que_chars)
		print('the number of chars for new answers:', res_chars)

	def main(self):
		if self.jingjian == '0':  
			if self.break_thre == -1:
				if self.break_thre4call != -1:
					print('Function: main_stopbythre2_jingjian')
					self.main_stopbythre2_jingjian()
				else:
					print('Function: main_all_pair_jingjian')
					self.main_all_pair_jingjian()
			else:
				print('Function: main_stopbythre_jingjian')
				self.main_stopbythre_jingjian()

		else:
			print('Unconsidered cases')


if __name__ == '__main__':

	mdopt = optparse.OptionParser()
	mdopt.add_option('-i', '--index', dest='index', type='int', default=-1)
	options, args = mdopt.parse_args()
	idx = options.index


	api = CluterbyApi()
	api.main()


