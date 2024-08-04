#coding:utf8
#直接调用llm api进行判断
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

sys.path.append('..')

from code4dbpedia.llm_api import TYQW_api
from code4dbpedia.llm_api import ChatGPT
from code4dbpedia.llm_api import LLama_api

class CluterbyApi(object):
	"""docstring for CluterbyApi"""
	def __init__(self, idx=-1, newprompt=9, break_thre=-1):
		self.dataset = 'dbpedia'
		self.dataset = 'cndbpedia'

		self.newprompt = newprompt   #记录prompt的类型，可选0，1，2
		self.break_thre = break_thre  #记录提前结束的阈值，3，5， -1， -1 表示不会提前结束
		self.break_thre4call = 200  #每个属性下固定只能调100次
		self.jingjian = '0'	#考虑相似的传递性，0考虑，1不考虑
		# self.jingjian = '1'
		self.target_idx = idx	#记录只处理一个属性时，它的下标
		
		path, _ = os.path.split(os.path.realpath(__file__))
		self.ci_flag = '0'  #考虑大小写
		
		if self.dataset == 'dbpedia':
			self.newprompt = 3   #prompt0是count  prompt是group prompt3是semantic prompt2是seman+3shot prompt1是seman+value prompt4是seman+1shot, prompt5是prompt1+prompt2
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_wp_tfidf_'+self.ci_flag+'.json')
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f2_0.json')
			self.target_attrcons = ['timeZone', 'architectureType', 'chairLabel', 
									'stat1Header', 'batting', 'areaBlank1Title', 
									'chrtitle', 'broadcastArea', 
									'sworntype', 'lakeType', 'scoreboard', 'timezone1Dst', 
									'link2Name']   #13个 'thirdRiderMoto2Country', 'membersLabel', 

			self.cleanres_file = self.dataset+'/stand_cleanres4' + self.dataset + '_ci_'+self.ci_flag+'.json'
			self.path = os.path.join(path, '../code4dbpedia')

		else:   #cndbpedia
			self.newprompt = 10   #3 是近义词， prompt10 是近义词+value,prompt8是近义词+3shot，prompt4是近似词+1shot , prompt7是同一+随机选的3shot, prompt11，value+3shot（prompt7）
			self.ci_flag = '0'  #中文的都是0
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_wp_tfidf_0.json')
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f6_0.json')
			self.target_attrcons = ['技术性质_车站', '国际濒危等级_学科', '设备种类_科技产品', '形象特征_人物', 
									'大坝类型_水电站', '成因类型_地貌', '所处时间_学科', '民居类型_地点', 
									'交建筑类型_建筑', '剧目类型_戏剧', '性别_人物', '行业类型_机构', '节目属性_娱乐', 
									'展馆类型_机构', '文章状态_小说']	#15
			# self.target_attrcons = ['性别_人物', '交建筑类型_建筑', '形象特征_人物', '成因类型_地貌', '大坝类型_水电站']
									
			self.cleanres_file = self.dataset+'/stand_cleanres4' + self.dataset + '_ci.json'
			self.path = os.path.join(path, '../canonlization')

		"""不同的prompt 0. 请问下面两个值算书籍的同一种类型吗？请回答是或不是。\n韩语\n外语
		1. 你是一名知识库质量管理者，正在对CNDBpedia这个知识库中的属性值进行聚类，对于小说状态这个属性下可能的取值有连载中、已完结、暂停更新、断更、TBC、预告/筹备中、草稿/试读版、修订中、废止/腰折、永久断更、定期更新、不定期更新、会员专享/付费章节、独家发布等，请问下面两个值算同一种小说状态吗？请回答是或不是。\n连载\n连载中
		2. 你是CNDBpedia知识库的质量管理者，正在对知识库中的属性值进行聚类，即给定某个属性下的两个取值，判断其是否是近义词，输出结果为“是”或“不是”。\n如：输入：属性“小说状态”下这两个取值是近义词吗？\n已完结\n完本\n输出：是\n输入：属性“小说状态”下这两个取值是近义词吗？\n未完结\n完本\n输出：不是\n输入：属性“小说状态”下这两个取值是近义词吗？\n更新中\n每天更新\n输出：是\n请问属性“小说状态”下这两个取值是近义词吗？
		"""
		if self.newprompt == 0:
			print('直接属性+概念提问，人工优化后的prompt')
			self.prompt_file = os.path.join(path, self.dataset, 'prompt0.json')
			self.read_prompts()
		elif self.newprompt == 1:
			print('利用新的prompt提问，人工设置的取值示例上下文')  #用人工设定的prompt
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt1.json')
			self.read_prompts()
		elif self.newprompt == 2:
			print('利用新的prompt提问，人工设置的3个判断示例上下文')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt2.json')
			self.read_prompts() 
		elif self.newprompt == 3:
			print('利用新的prompt提问')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt3.json')
			self.read_prompts() 
		elif self.newprompt == 4:
			print('利用新的prompt提问，只有一个示例')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt4.json')
			self.read_prompts() 
		elif self.newprompt == 5:
			print('利用新的prompt提问，人工设置的3个判断示例上下文，和dbpedia一致')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt5.json')
			self.read_prompts() 
		elif self.newprompt == 6:
			print('利用旧的prompt提问，人工设置的3个判断示例上下文，和dbpedia一致')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt6.json')
			self.read_prompts()
		elif self.newprompt == 7:
			print('利用旧的prompt提问，随机设置的3个判断示例上下文，和dbpedia一致')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt7.json')
			self.read_prompts() 
		elif self.newprompt == 8:
			print('利用prompt3提问，人工设置的3个判断示例上下文，和dbpedia一致')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt8.json')
			self.read_prompts() 
		elif self.newprompt == 9:
			print('请问下面两个值在属性“居民类型”下这两个取值是近似词吗？请回答是或不是。')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt9.json')
			self.read_prompts() 
		elif self.newprompt == 10:
			print('先给可能取值列表，请问下面两个值在属性“居民类型”下这两个取值是近似词吗？请回答是或不是。')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt10.json')
			self.read_prompts()
		elif self.newprompt == 11:
			print('先给可能取值列表，请问下面两个值算科技产品的同一种设备种类吗？请回答是或不是。+3shot')
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt11.json')
			self.read_prompts()
		elif self.newprompt == -1:
			print('最开始的prompt')
			self.prompt_file = os.path.join(path, self.dataset, 'prompt.json')
			self.read_prompts() 
		else:
			print('无效prompt, cndbpedia只有三种prompt', self.prompt)
			exit()
		
		if self.target_idx >= 0:					
			self.target_attrcons = [self.target_attrcons[self.target_idx]]
		# else:
		# 	self.target_attrcons = []	
		
		print('target_idx:', self.target_idx, len(self.target_attrcons))	
		print('当前处理的属性是：', self.target_attrcons)

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
			pass

		if self.newprompt >= 0:
			# idx = -2   #-2表示prompt3重新用gpt3.5-turbo跑
			# idx = -4  # -4表示prompt3重新用gpt3.5-turbo-0125跑
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
			print('文件已存在，需重新设置idx值', self.temp_llm_res_file)
			exit()
		if os.path.exists(self.llm_add_res_file):
			print('文件已存在，需重新设置idx值', self.llm_add_res_file)
			exit()

		# start = time.time()	
		# res = self.llm_api.get_response("请问下面两个值算人物的同一种形象特征吗？请回答是或不是。\n以蝴蝶为形象特征\n干练")
		# print(res)
		# print(time.time()-start)
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
			data = json.load(fr)
		# data = {}
		keys = list(data.keys())
		if self.dataset == 'cndbpedia':
			for k in keys:
				for v in list(data[k].keys()):
					if k+'_'+v not in self.target_attrcons:
						del data[k][v]
						if not data[k]:
							del data[k]

		elif self.dataset == 'dbpedia':
			for k in keys:
				# info = data[k]
				if k not in self.target_attrcons:
					del data[k]

		return data

	def read_sims(self):
		# print('sim_res_file:', self.sim_res_file)
		all_sim_pairs = {}
		with open(self.sim_res_file, 'r', encoding='utf-8') as fr:
			line = fr.readline()
			while line:
				if '当前处理的属性值' in line:
					attr = fr.readline().strip()
					line = fr.readline()
					con = fr.readline().strip()
					line = fr.readline()
					# print('相似度计算结果:', attr, con, line[:100])
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
			print('已有大模型判断的结果', self.llm_res_file)
			with open(self.llm_res_file, 'r', encoding='utf8') as fr:
				llm_res = json.load(fr)

			# keys = list(llm_res.keys())
			# if self.dataset == 'cndbpedia':
			# 	for attr in keys:
			# 		for con in list(llm_res[attr].keys()):
			# 			if attr +'_'+con not in self.target_attrcons:
			# 				del llm_res[attr][con]
			# 		if not llm_res[attr]:
			# 			del llm_res[attr]

			# elif self.dataset == 'dbpedia':
			# 	for attr in keys:
			# 		if attr not in self.target_attrcons:
			# 			del llm_res[attr]			
		else:
			print('文件不存在', self.llm_res_file)
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
			ans = ans.split('output')[-1]   #英文
			ans = ans.split('输出')[-1]     #中文
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
				# print('--------------\n没有输出对应的答案', i)
				# print('v0:', v0)
				# print('v1:', v1)
				# continue

			else:   #有多个也只取第一个返回结果
				ans1 = target_res[0].split('Output')[-1].strip(': \n：')
				ans1 = ans1.split('\n')[0]
				if ans1 in ['yes']:
					res = 'yes'
				elif ans1 in ['no']:
					res = 'no'
				else:
					res = 'unsure'
					# print('*******************\n未考虑到的情况', i)
					# print('v0: ', v0)
					# print('v1:', v1) 
					# print('ans1:', ans1)
					
		return res

	def post_no(self, temp_res, all_data):
		#后处理聚类结果，不考虑传递性，一个簇中的所有候选对判断为yes	
		#temp_res中保存的是候选对, [[],[]]
		#all_data 所有待处理的值 ,[]
		# a,b,c 三个结果都是yes才会被聚集在一起
		mappings = defaultdict(set)
		all_cs = []

		for x in temp_res:
			v0, v1 = x
			mappings[v0].add(v1)
			mappings[v0].add(v0)
			mappings[v1].add(v1)
			mappings[v1].add(v0)

		# print(mappings)

		all_vs = list(mappings.keys())

		temp_res = []
		final_res = []
		added_res = []   #已经被考虑过的组合
		for i in range(len(all_vs)-1):
			x0 = all_vs[i]
			syn_x0 = mappings[x0]
			# print('x0:', x0)
			# for j in range(i+1, len(all_vs)):
			# 	x1 = all_vs[j]
			# 	if x1 not in syn_x0:
			# 		continue
			for x1 in mappings[x0]:
				same = mappings[x1] & syn_x0
				if x0 not in same:
					continue
				elif x1 not in same:
					continue
				if len(same) == 2:
					final_res.append(same)
				else:
					if len(same) > 2:
						temp_res.append([same, same-set([x0, x1]), 2])
						added_res.append(same)

		def inner(temp_c, mappings, final_res, added_res):
			# temp_c :[same, remain, num]   #same当前找到的语义相同的值，remain：same中还没有确认的候选值,num已经确认过的值的个数
			temp_res = []
			# print('temp_c:', temp_c)
			for x in temp_c[1]:
				y = mappings[x]
				same = temp_c[0] & y
				if x not in same:
					continue 
				if len(same) == temp_c[2]+1:
					final_res.append(same)
				elif len(same) > temp_c[2]+1:
					# if len(temp_c[1]) == 1:
					# 	final_res.append(same)
					# else:
					# 	if not (same & (temp_c[1]-set([x]))):
					# 		final_res.append(same)
					# 		continue
					flag = True
					for k in added_res:
						if k == same:
							flag = False
							break
					if not flag:
						continue

					for i, k in enumerate(temp_res):
						if same == k[0]:
							flag = False
							break
						elif not(k[0]-same) and (same-k[0]):   #same包含k
							flag= False
							break
						elif not(same-k[0]) and (k[0]-same):  #k包含same
							temp_res[i] = [same, temp_c[1]-set([x]), temp_c[2]+1]
							added_res.append(same)
							flag=False
							break
					# for k in final_res:
					# 	if (same - k) and not (k -same)		
					if flag:
						try:
							temp_res.append([same, temp_c[1]-set([x]), temp_c[2]+1])
						except:
							print("temp_res:", temp_res)
							print("same:", same)
							print("temp_c:", temp_c)
							print('x:', x)
							exit()
						added_res.append(same)
						# temp_res.append([same, temp_c[1]-set([x])])
			return temp_res

		# print("final_res:", final_res)
		# print('------------1\ntemp_res:', temp_res)
		
		count = 1
		while True:
			mid_res = []
			for x in temp_res:
				mid_res.extend(inner(x, mappings, final_res, added_res))
			if not mid_res:
				break
			temp_res = copy.deepcopy(mid_res)
			count += 1
			# print('------------', count+1, '\ntemp_res:', temp_res)
			# print("final_res:", final_res)
			if count > 500:
				print('未被处理的:',)
				print('temp_res:', temp_res)
				break
		res = []
		for x in final_res:
			# print('--------------\n', res)
			flag = True
			for i, y in enumerate(res):
				# if (not (x-y)) and (not (y-x)):
				if x == y:
					flag = False
					break
				if (not(y-x) ) and (x-y):
					res[i] = x
					flag=False
					break
				if (not (x-y)) and (y-x):
					flag=False
					break
			if flag:
				res.append(x)
			# print(res)

		final_res = []
		added_values = set()
		for x in res:
			final_res.append(list(x))
			added_values |= x
		for x, y in all_data.items():
			if x not in added_values:
				final_res.append([x])
		return final_res		

	def post(self, temp_res, all_data):
		#后处理聚类结果，具有传递性	
		#temp_res中保存的是候选对
		#all_data 所有待处理的值
		mapping = {}  #记录每个值所在的簇idx
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
		"""所有的候选值对都判断一次，
		   判断时不考虑传递性，但聚簇时ac所在的簇会直接被合并，即如果a与b相似，a与c相似，直接默认b和c相似，
		"""
		data = self.get_data()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #调用api的次数统计
		api_temp = 0   #新调用的api次数
		que_chars = 0   #调用api的问题的字符数
		res_chars = 0   #api回答的字符数
		for attr, info in data.items():
			if api_temp > 5000:
				break
			print('------------------\n当前处理的属性：', attr)
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
					print('-----------当前处理的概念：', con)
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					mappings = defaultdict(list)   #记录每个值可能在的簇id
					all_vs = list(c_info.keys())
					print('所有候选值个数:', len(all_vs))
					for i in range(len(all_vs)-1):
						v0 = all_vs[i].split('#****#')[0]
						added_idx = len(temp_final_res)   #每个轮次只能新增，不可能存在重组
						for j in range(i+1, len(all_vs)):
							v1 = all_vs[j].split('#****#')[0]
							#先判断是否已经在一个簇
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
								# print('p已有结果', p, ans)
								if ans in ['no', 'unsure', 'failure', 'unknow']:
									continue
							else:
								if v1.lower() == v0.lower():
									ans = 'yes'	
								elif p1 in temp_llm_res:
									ans = temp_llm_res[p1]
									# print('p已有结果', p, ans)
									if ans in ['no', 'unsure', 'unknow']:   #不成簇，直接跳到下一候选对
										continue
								else:
									q = prompt+v0+'\n'+v1
									que_chars += len(q)
									temp = [attr, q, con, v0, v1]

									api_temp += 1
									print(v0, v1, '没有llm结果')
									continue
									# i = random.randint(0,1)
									# if i == 0:
									# 	ans = 'yes'
									# 	res = 'yes'
									# else:
									# 	ans = 'no'
									# 	res = 'no'

									# res = self.llm_api.get_response(q)
									# res_chars += len(res)
									# ans = self.get_llm_res(res)

									# temp.insert(2, ans)   # 处理后的结果
									# temp.append(res)	 #处理前的结果
									
									if ans in ['failure']:
										continue

									added_llm_res.append(temp)
									temp_llm_res[p] = ans 	
									# break
									if ans in ['no', 'unsure', 'unknow']:
										continue

							# print('新增簇:', p)
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
		print('一共调用api：', api_total, '新调用的api次数：', api_temp)
		if added_llm_res:
			print('added_llm_res写入文件', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	

			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)				

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)		

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)


		print('新增的问题字符数:', que_chars)
		print('新增问题的答案字符数：', res_chars)
	
	def main_stopbythre_jingjian(self):
		#利用阈值提前结束，并利用可传递性对大模型的调用次数进行精简
		data = self.get_data()
		all_sim_pairs = self.read_sims()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #调用api的次数统计
		api_temp = 0   #新调用的api次数
		que_chars = 0   #调用api的问题的单词数
		res_chars = 0   #api回答的单词数
		# all_chars = 0   #所有需要调用的问题的字符数

		for attr, info in data.items():
			if api_temp > 8000:
				break
			print('------------------\n当前处理的属性：', attr)
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
					print('-----------当前处理的概念：', con)
					sims = all_sim_pairs[attr][con]
					sims = sorted(sims.items(), key=lambda x:x[1], reverse=True)
					mappings = defaultdict(list)   #记录每个值可能在的簇id
					all_vs = list(c_info.keys())
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					temp_final_res = defaultdict(list)
					no_count = 0   #继续连续多少个被判为no
					temp_apicount = 0   #当前属性下调api的次数
					for (pair, v) in sims:
						v0 = pair[0].split('#****#')[0]
						v1 = pair[1].split('#****#')[0]
						added_idx = len(temp_final_res)   #每个轮次只能新增，不可能存在重组
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
								# print(v0, v1, '没有llm结果')
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
									temp.insert(2, ans)   # 处理后的结果
									temp.append(res)	 #处理前的结果
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
								print('结束', pair, v, ans)
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
		print('一共调用api：', api_total, '新调用的api次数：', api_temp)
		if added_llm_res:
			print('added_llm_res写入文件', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	
			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)					

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)	

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)

		print('新增的问题字符数:', que_chars)
		print('新增问题的答案字符数：', res_chars)

	def main_stopbythre2_jingjian(self):
		#利用阈值提前结束，并利用可传递性对大模型的调用次数进行精简
		data = self.get_data()
		all_sim_pairs = self.read_sims()
		llm_res = self.read_llm_res()

		final_res = {}
		mid_res = {}
		added_llm_res = []
		api_total = 0 #调用api的次数统计
		api_temp = 0   #新调用的api次数
		que_chars = 0   #调用api的问题的单词数
		res_chars = 0   #api回答的单词数
		# all_chars = 0   #所有需要调用的问题的字符数

		for attr, info in data.items():
			print('------------------\n当前处理的属性：', attr)
			final_res[attr] = {}
			mid_res[attr] = {}
			for con, c_info in info.items():
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					print('-----------当前处理的概念：', con)
					sims = all_sim_pairs[attr][con]
					sims = sorted(sims.items(), key=lambda x:x[1], reverse=True)
					mappings = defaultdict(list)   #记录每个值可能在的簇id
					all_vs = list(c_info.keys())
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					temp_final_res = defaultdict(list)
					no_count = 0   #继续连续多少个被判为no
					temp_apicount = 0   #当前属性下调api的次数
					for (pair, v) in sims[:self.break_thre4call]:   #只考虑前100个
						v0 = pair[0].split('#****#')[0]
						v1 = pair[1].split('#****#')[0]
						added_idx = len(temp_final_res)   #每个轮次只能新增，不可能存在重组
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
								# print(v0, v1, '没有llm结果')
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
									temp.insert(2, ans)   # 处理后的结果
									temp.append(res)	 #处理前的结果
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
								print('结束', pair, v, ans)
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
		print('一共调用api：', api_total, '新调用的api次数：', api_temp)
		if added_llm_res:
			print('added_llm_res写入文件', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
				json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	
			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(llm_res, fw, ensure_ascii=False, indent=4)					

		with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
			json.dump(mid_res, fw,  ensure_ascii=False, indent=4)	

		with open(self.final_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)

		print('新增的问题字符数:', que_chars)
		print('新增问题的答案字符数：', res_chars)

	def main_stopbythre(self):
		#利用阈值提前结束，不考虑传递性
		data = self.get_data()
		all_sim_pairs = self.read_sims()
		llm_res = self.read_llm_res()

		final_res = {}
		added_llm_res = []
		api_total = 0 #调用api的次数统计
		api_temp = 0   #新调用的api次数
		que_chars = 0   #调用api的问题的字符数
		res_chars = 0   #api回答的字符数
		start_time = time.time()
		for attr, info in data.items():
			if api_temp > 4000:
				break
			print('------------------\n当前处理的属性：', attr)
			final_res[attr] = {}
			for con, c_info in info.items():
				if api_temp > 4000:
					break
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					print('-----------当前处理的概念：', con)
					sims = all_sim_pairs[attr][con]
					sims = sorted(sims.items(), key=lambda x:x[1], reverse=True)
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					temp_res = []	#记录判断结果为相同的值对
					no_count = 0   #继续连续多少个被判为no
					for (pair, v) in sims:
						api_total += 1
						v0 = pair[0].split('#****#')[0]
						v1 = pair[1].split('#****#')[0]
						if self.dataset == 'cndbpedia':
							p = v0 +'#****#'+ v1
							p1 = v1 +'#****#'+ v0
						elif self.dataset == 'dbpedia':
							p = v0.lower() +'#****#'+ v1.lower()	
							p1 = v1.lower() +'#****#'+ v0.lower()

						if v0.lower() == v1.lower():
							ans = 'yes'

						elif p in temp_llm_res:
							ans = temp_llm_res[p] 
							
						elif p1 in temp_llm_res:
							ans = temp_llm_res[p1] 
				
						else:
							q = prompt+v0+'\n'+v1
							que_chars += len(q)

							temp = [attr, q, con, v0, v1]
							api_temp += 1
							# print(v0, v1, '没有llm结果')
							i = random.randint(0,1)
							if i == 0:
								ans = 'yes'
								res = 'yes'
							else:
								ans = 'no'
								res = 'no'

							# continue

							# res = self.llm_api.get_response(q)
							# res_words += len(res)
							# ans = self.get_llm_res(res)
							if ans not in ['failure']:
								temp.insert(2, ans)   # 处理后的结果
								temp.append(res)	 #处理前的结果
								added_llm_res.append(temp)
								temp_llm_res[p] = ans

						if ans in ['yes']:
							no_count = 0
							temp_res.append(pair)
							# temp_res.append([v0, v1])
						elif ans in ['no']:
							no_count += 1
							if no_count == self.break_thre:
								print('结束', pair, v, ans)
								break
					
					llm_res[attr][con] = temp_llm_res	
						# break
						# print(pair, ans)
					# print('当前结果：', temp_res)
					# print("c_info:", c_info)
					final_res[attr][con] = self.post_no(temp_res, c_info)
					# print('###########3\n聚类结果：', final_res[attr][con])
				except:
					traceback.print_exc()

				break
			break
		end_time = time.time()
		# print("added_llm_res:", added_llm_res)
		# print('一共调用api：', api_total, '新调用的api次数：', api_temp)
		# if added_llm_res:
		# 	with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
		# 		json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)					
		# 	with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
		# 		json.dump(llm_res, fw, ensure_ascii=False, indent=4)
					
		# with open(self.final_res_file, 'w', encoding='utf8') as fw:
		# 	json.dump(final_res, fw, ensure_ascii=False, indent=4)
		print('使用时间：', end_time - start_time)
		print('新增的问题单词数:', que_chars)
		print('新增问题的答案单词数：', res_chars)

	def main_all_pair(self):
		#所有候选对都判断一遍，当一个簇内所有的要素都互相yes时这个簇才成立
		data = self.get_data()
		llm_res = self.read_llm_res()

		final_res = {}   #最终聚类结果
		mid_res = {}   #中间结果，llm返回的结果
		added_llm_res = []
		api_total = 0 #调用api的次数统计
		api_temp = 0   #新调用的api次数
		que_chars = 0   #调用api的问题的字符数
		res_chars = 0   #api回答的字符数

		for attr, info in data.items():
			print('------------------\n当前处理的属性：', attr)
			if api_temp > 2000:
				break

			final_res[attr] = {}
			mid_res[attr] = {}
			for con, c_info in info.items():
				if api_temp > 2000:
					break
				try:
					if self.dataset == 'dbpedia':
						prompt = self.new_prompts[attr]
					elif self.dataset == 'cndbpedia':
						prompt = self.new_prompts[attr+'_'+con]
					final_res[attr][con] = []
					print('-----------当前处理的概念：', con)
					temp_llm_res = llm_res.get(attr, {}).get(con, {})
					# mappings = defaultdict(list)   #记录每个值可能在的簇id
					yes_pairs = []   #判断相同的候选对， list形式
					yes_list = [] #判断相同的候选对,string形式
					all_vs = list(c_info.keys())
					print('所有候选值个数:', len(all_vs))
					# all_vs = all_vs[:10]
					for i in range(len(all_vs)-1):
						v0 = all_vs[i].split('#****#')[0]
						for j in range(i+1, len(all_vs)):
							v1 = all_vs[j].split('#****#')[0]
								
							api_total += 1
							k = v0.lower() +'#****#'+ v1.lower()
							if k in temp_llm_res:
								ans = temp_llm_res[k] 
							else:
								k = v1.lower()+'#****#'+v0.lower()
								if k in temp_llm_res:
									ans = temp_llm_res[k] 	
								elif v1.lower() == v0.lower():
									ans = 'yes'							
								else:
									q = prompt+v0+'\n'+v1
									que_chars += len(q.split(' '))
									# print(q)
									temp = [attr, q, con, v0, v1]
									api_temp += 1

									i = random.randint(0,1)
									if i == 0:
										ans = 'yes'
										res = 'yes'
									else:
										ans = 'no'
										res = 'no'
									res_chars += len(res)
									# res = self.llm_api.get_response(q)
									# ans = self.get_llm_res(res)
									
									if ans not in ['failure']:
										temp.insert(2, ans)   # 处理后的结果
										temp.append(res)	 #处理前的结果
										added_llm_res.append(temp)
										temp_llm_res[k] = ans

							if ans in ['yes']:
								yes_pairs.append([v0, v1])
								yes_list.append(v0+'_'+v1)

					llm_res[attr][con] = temp_llm_res
					mid_res[attr][con] = yes_pairs
				except:
					traceback.print_exc()

				try:
					trans = {}
					for x in c_info.keys():
						new_x = x.split('#****#')[0]
						trans[new_x] = x

					added_vs = set()
					temp_pairs = []
					added_idx = set()
					print('yes_pairs：', yes_pairs)
					print('yes_list:', yes_list)

					for i in range(len(yes_pairs)):
						c_i = yes_pairs[i]
						for j in range(i+1, len(yes_pairs)):
							c_j = yes_pairs[j]  

							flag = True
							for x in c_i:
								for y in c_j:
									if x == y:
										continue
									if (x + '_' + y not in yes_list) and (y + '_' + x not in yes_list):
										flag = False
										break
								if not flag:
									break
							if flag:
								c_i = list(set(c_i) | set(c_j))
								added_idx.add(j)

						if i not in added_idx:
							temp_pairs.append(c_i)

					for pair in temp_pairs:
						temp_added_vs = set()  
						for x in pair:  
							temp_added_vs.add(trans[x])
						added_vs |= temp_added_vs

						final_res[attr][con].append(list(temp_added_vs))
					
					# print('added_vs:', added_vs)
					# print('all_vs；', all_vs[:10])
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
		print('一共调用api：', api_total, '新调用的api次数：', api_temp)
		# if added_llm_res:
		# 	with open(self.llm_add_res_file, 'w', encoding='utf8') as fw:
		# 		json.dump(added_llm_res, fw, ensure_ascii=False, indent=4)	
		# 	with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
		# 		json.dump(llm_res, fw, ensure_ascii=False, indent=4)					

		# with open(self.final_res_file, 'w', encoding='utf8') as fw:
		# 	json.dump(final_res, fw, ensure_ascii=False, indent=4)

		# with open(self.temp_final_res_file, 'w', encoding='utf8') as fw:
		# 	json.dump(mid_res, fw,  ensure_ascii=False, indent=4)

		print('新增的问题字符数:', que_chars)
		print('新增问题的答案字符数：', res_chars)

	
	def main(self):
		if self.jingjian == '0':  #有传递性
			if self.break_thre == -1:
				if self.break_thre4call != -1:
					print('调用的函数： main_stopbythre2_jingjian')
					self.main_stopbythre2_jingjian()
				else:
					print('调用的函数： main_all_pair_jingjian')
					self.main_all_pair_jingjian()
			else:
				print('调用的函数： main_stopbythre_jingjian')
				self.main_stopbythre_jingjian()

		elif self.jingjian == '1':  #没有传递性
			if self.break_thre == -1:
				print('调用的函数： main_all_pair')
				self.main_all_pair()
			else:
				print('调用的函数： main_stopbythre')
				self.main_stopbythre()

		else:
			print('未考虑到的情况')
		# if self.newprompt:
		# 	self.main_w_newp_jingjian()

		# elif self.break_thre > 0:
		# 	print('根据阈值提前结束判断流程')
		# 	# self.main_stopbythre()
		# 	self.main_stopbythre_jingjian()
		# else:
		# 	self.main_all_pair()


# class Merge:
# 	def __init__(self):
# 		pass 

# 	def merge_res(self):
# 		suffix = ['_0', '_1', '_2']
# 		prefix = '/mnt/heying/common_code/dbpedia/qwen_max_res_large_0_-1'
# 		new_file = prefix +'.json'
# 		print('保存的新文件：', new_file)
# 		all_res = {}
# 		for x in suffix:
# 			file = prefix + x+'.json'
# 			print('当前处理的文件：', file)
# 			with open(file, 'r', encoding='utf8') as fr:
# 				temp = json.load(fr)
# 				for k in temp:
# 					all_res[k] = temp[k]

# 		print('一共处理的属性个数：', len(all_res))

# 		with open(new_file, 'w', encoding='utf8') as fw:
# 			json.dump(all_res, fw, ensure_ascii=False, indent=4)


if __name__ == '__main__':
	# mer = Merge()
	# mer.merge_res()

	mdopt = optparse.OptionParser()
	mdopt.add_option('-i', '--index', dest='index', type='int', default=-1)
	options, args = mdopt.parse_args()
	idx = options.index


	api = CluterbyApi()
	# api.main_stopbythre()
	# api.main_w_newp()
	# api.main_all_pair()
	api.main()


