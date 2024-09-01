#coding:utf8
#correct with LLM

import json
import os, sys
import traceback
import re
from collections import defaultdict, Counter
import numpy as np
from copy import deepcopy
import random

random.seed(43)


from llm_api import TYQW_api
from llm_api import ChatGPTKey
from llm_api import Baichuan_api
from llm_api import LLama_api

from final_cluster_by_api import CluterbyApi
clusterapi = CluterbyApi()


class Post:
	def __init__(self):
		print('-----------------------------------\n-----------------------------------------begin')
		self.dataset = 'dbpedia'   
		# self.dataset = 'cndbpedia'

		self.llm_model = 'gpt35'
		# self.llm_model = 'qwen_max'
		# self.llm_model = 'baichuan'
		# self.llm_model = 'llama'
		# self.llm_model = 'llama2'
		self.llm_model = 'llama3'
		self.ci_flag = '1'
		self.ci_flag = '0'

		if self.llm_model == 'qwen_max':
			model = 'qwen-max-0403'
			self.llm_api = TYQW_api(model)
		elif self.llm_model == 'gpt35':
			model = 'gpt-3.5-turbo'
			self.llm_api = ChatGPT(model)
		elif self.llm_model == 'baichuan':
			model = 'Baichuan3-Turbo'
			self.llm_api = Baichuan_api(model)
		elif self.llm_model == 'llama':
			model = 'llama_3_8b'
			self.llm_api = LLama_api(model)
		elif self.llm_model == 'llama2':
			model = 'llama_2_13b'
			self.llm_api = LLama_api(model)
		elif self.llm_model == 'llama3':
			model = 'llama3_70b'
			self.llm_api = LLama_api(model)

		path, _ = os.path.split(os.path.realpath(__file__))

		if self.dataset == 'dbpedia':
			self.p_suffix = '\nOutput:'
			self.newprompt = 2   # the best prompt 
			idx = 0
			self.cufen_res_file = os.path.join(path, 'baselines', 'our_dbpedia_text_emb_large_cp_w_f20.8_0.json')   
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f2_0.json')     
			self.target_attrcons = ['timeZone', 'architectureType', 'chairLabel', 
									'stat1Header', 'batting', 'areaBlank1Title', 
									'chrtitle', 'broadcastArea', 
									'sworntype', 'lakeType', 'scoreboard', 'timezone1Dst', 
									'link2Name']   #13 
		elif self.dataset == 'cndbpedia':
			self.p_suffix = ''
			self.newprompt = -1 #the best prompt 
			idx = 3
			self.cufen_res_file = os.path.join(path, 'baselines' ,'our_cndbpedia_text_emb_large_cp_w_f60.75.json')	
			self.sim_res_file = os.path.join(path, self.dataset, 'sim_pairs_file_text_emb_large_cp_w_f6_0.json')
			self.target_attrcons = ['技术性质_车站', '国际濒危等级_学科', '设备种类_科技产品', '形象特征_人物', 
									'大坝类型_水电站', '成因类型_地貌', '所处时间_学科', '民居类型_地点', 
									'交建筑类型_建筑', '剧目类型_戏剧', '性别_人物', '行业类型_机构', '节目属性_娱乐', 
									'展馆类型_机构', '文章状态_小说']
								
		self.target_attrcons = None	  # if target_attrcons = None means correction for all attributes

		if self.newprompt >= 0:
			if self.dataset == 'dbpedia':
				idx = 1
			elif self.dataset == 'cndbpedia':
				idx = 2
			self.llm_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_all_q_ans_newp'+str(self.newprompt)+'_'+str(idx)+'.json')  
			self.llm_add_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_added_q_res_newp'+str(self.newprompt)+'_'+str(idx)+'.json')
			self.temp_llm_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_all_q_ans_newp'+str(self.newprompt)+'_'+str(idx+1)+'.json')
			self.post_res_file = self.cufen_res_file.replace('.json', '_post_'+self.llm_model+'_p'+str(self.newprompt)+'_15_intra.json')
		
		else:
			idx = 9
			if self.llm_model == 'gpt35':
				if self.dataset == 'cndbpedia':
					tag = '03m_'
				elif self.dataset == 'dbpedia':
					tag = '02m_'
			else:
				tag = ''
			self.llm_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_all_questions_ans_'+tag+str(idx)+'.json')  
			self.llm_add_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_added_q_res_'+tag+str(idx)+'.json')
			self.temp_llm_res_file = os.path.join(path, 'final', self.llm_model, self.dataset+'_'+self.llm_model+'_all_questions_ans_'+tag+str(idx+1)+'.json')
			self.post_res_file = self.cufen_res_file.replace('.json', '_post_'+self.llm_model+'_15_intra.json')

		if os.path.exists(self.temp_llm_res_file):
			print('The file already exists. You need to reset the idx', self.temp_llm_res_file)
			exit()
		if os.path.exists(self.llm_add_res_file):
			print('The file already exists. You need to reset the idx', self.llm_add_res_file)
			exit()
				
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
		elif self.newprompt == -1:
			self.prompt_file = os.path.join(path, self.dataset, 'prompt.json')
			self.read_prompts() 
		elif self.newprompt == 7:
			self.prompt_file = os.path.join(path, self.dataset, 'new_prompt7.json')
			self.read_prompts() 
		else:
			print('unknow prompt templates', self.newprompt)
			exit()
		
		self.tosupple_pairs_num = 0
		self.tosupple_pairs = {}
		self.needed_pairs = 0
		self.attr = ''
		self.con = ''
		self.res_chars = 0    
		self.que_chars = 0   
		self.added_llm_res = [] 
		print('dataset:', self.dataset)
		print('newprompt:', self.newprompt)
		print('LLM:', self.llm_model, 'ci_flag:', self.ci_flag)
		print('llm_res_file:', self.llm_res_file)
		print('temp_llm_res_file:', self.temp_llm_res_file)
		self.intra_ave_sims = {}
		self.intra_min_sims = {}
		self.inter_min_sims = {}
		self.inter_ave_sim = {}
		self.ave_sims = {}
		self.intra_pair_sims = {}
		self.inter_pair_sims = {}

	def read_prompts(self):
		with open(self.prompt_file, 'r', encoding='utf-8') as fr:
			self.new_prompts = json.load(fr)	

	def get_center(self, gather, sim_pairs):
		len_  = len(gather)   
		sims = []   
		sum_sim = 0
		if len_ in [1]:
			return {'center': gather[0]} #} #, 'ave_sim': 0.0 
		elif len_ in [2]:
			k = (gather[0].split('#****#')[0], gather[1].split('#****#')[0])
			if k in sim_pairs:
				ave_sim = sim_pairs[k]
			else:
				k = (gather[1].split('#****#')[0], gather[0].split('#****#')[0])
				if k in sim_pairs:
					ave_sim = sim_pairs[k]
				else:
					ave_sim = 0.0
					print(k, 'the pair has no similarity')
			return {'center': gather[0], 'all_ave_sim': ave_sim, 'ave_sim': ave_sim, 'min_sim':ave_sim, 'sim_matrix': [[1.0, ave_sim], [ave_sim, 1.0]]}
					
		for i in range(len_):
			sims.append([0]*len_)

		for i in range(len_):
			sims[i][i] = 1.0
			for j in range(i+1, len_):
				k = (gather[i].split('#****#')[0], gather[j].split('#****#')[0])
				if k in sim_pairs:
					sims[i][j] = sim_pairs[k]
					sims[j][i] = sims[i][j]
				else:
					k = (gather[j].split('#****#')[0], gather[i].split('#****#')[0])
					if k in sim_pairs:
						sims[i][j] = sim_pairs[k]
						sims[j][i] = sims[i][j]
					else:
						print(k, 'the pair has no similarity')
				sum_sim += sims[i][j]
		# print("get_center sims:", sims)		
		ave_sim = sum_sim / (len_*(len_-1)/2.0)   #the average similarity of all point pairs in the cluster
		# the point has the maximum similarity with other points in the same cluster is the center of the cluster
		max_idx = 0
		max_sim = 0
		min_sim = 1.0
		for i in range(len_):
			sum_ = sum(sims[i])
			min_sim = min(min_sim, min(sims[i]))  
			if sum_ > max_sim:
				max_idx = i
				max_sim = sum_
		ave_sim2c = round(np.mean(sims[max_idx][:max_idx]+sims[max_idx][max_idx+1:]), 4)    #the mean similarity of the center point with other points in the cluster 

		if min_sim == 1.0 and ave_sim2c != 1.0:
			print('there is a error in this process:', ave_sim2c, sims )
		return {'center': gather[max_idx], 'all_ave_sim': ave_sim, 'min_sim':min_sim, 'sim_matrix': sims, 
				'center_idx':max_idx, 'ave_sim': ave_sim2c}

	def get_intra_cand_pairs(self, data, ave_sim):
		"""
		find low confidence point within a cluster
		args:
			data: {center: {'values':list(values), ave_sim:float(), sim_matrix}}} 
			ave_sim: the mean similarity of all clusters
		returns:
			cand_pairs:{center:[]}
		"""
		cand_pairs = {}
		for k, info in data.items():
			try:
				values = info['values']
				len_ = len(values)
				if len_ == 1:
					continue

				cand_pairs[k] = []	
				intra_ave_sim = info['ave_sim']    
				sim_matrix = info['sim_matrix']

				center_idx = values.index(k)
				print('current cluster:', k, 'ave_sim:', ave_sim, 'intra_ave_sim:', intra_ave_sim)
				print('candidate similarity:', np.array(sim_matrix)[:, center_idx])
				for i in range(len_):
					if i == center_idx:
						continue

					sim = sim_matrix[i][center_idx]
					if sim < ave_sim or sim < intra_ave_sim:
						cand_pairs[k].append([values[i], values[center_idx]])
			except:
				traceback.print_exc()
				print(k, info)			
		return cand_pairs

	def get_llm_anss(self, cand_pairs, llm_res):
		"""
		args:
			cand_pairs:{center:[]}
			llm_res: {p:str}
		returns:
			new_res: {center:{p:int}}
		"""
		new_res = {}
		temp_api = 0  #current calling number of LLMs
		for k, all_pairs in cand_pairs.items():
			temp_api += len(all_pairs)
		if temp_api > 2000:
			for k, all_pairs in cand_pairs.items():
				n = len(all_pairs)
				n = int((2000/temp_api)*n)
				all_pairs = random.sample(all_pairs, n)
				cand_pairs[k] = all_pairs

		for k, all_pairs in cand_pairs.items():
			new_res[k] = {}
			
			self.needed_pairs += len(all_pairs)
			
			for pair in all_pairs:
				v0 = pair[0].split('#****#')[0]
				v1 = pair[1].split('#****#')[0]
				if self.dataset == 'dbpedia':
					p = v0.lower()+'#****#'+v1.lower()
					p1 = v1.lower()+'#****#'+v0.lower()
				else:
					p = v0 + '#****#' + v1
					p1 = v1 + '#****#' + v0

				if p in llm_res:
					new_res[k][pair[0]+'--##--'+pair[1]] = llm_res[p]
				else:

					# p = pair[1].split('#****#')[0].lower()+'#****#'+pair[0].split('#****#')[0].lower()
					if p1 in llm_res:
						new_res[k][pair[0]+'--##--'+pair[1]] = llm_res[p1]
					else:
						print('candidate pair that did not judged by LLMs', p)
						self.tosupple_pairs[self.attr][self.con].append(pair)
						self.tosupple_pairs_num += 1
						# continue
						if v1.lower() == v0.lower():
							ans = 'yes'	
						else:
							q = self.prompt + v0 + '\n' + v1 + self.p_suffix
							self.que_chars += len(q)
							temp = [self.attr, q, self.con, v0, v1]

							res = self.llm_api.get_response(q)
							self.res_chars += len(res)
							ans = clusterapi.get_llm_res(res, llama=True, v0=v0, v1=v1, q=q)

							temp.insert(2, ans)  
							temp.append(res)	 
							self.added_llm_res.append(temp)
							self.llm_data[self.attr][self.con][p] = ans

							# print(temp)
							# print(p, res, ans)
							# exit()

						new_res[k][pair[0]+'--##--'+pair[1]] = ans


		print('judgements from LLMs:', new_res)
		return new_res

	def clean_clus_by_llm(self, old_clus, llm_res):
		"""cleaning the cluster by the results from LLMs
		args:
			old_clus: {center:{'values':list, 'center':str, ....}}
			llm_res: {center:{p:str}
		returns:
			new_clus: {center:[]}  #new clusters
			noadded: list #values removed from old clusters
		"""	
		noadded = []
		new_clus = {}
		for k, info in old_clus.items():
			new_clus[k] = info['values']
			llm_info = llm_res.get(k, {})
			for p, res in llm_info.items():
				if 'no' not in res:
					continue
				vs = p.split('--##--')
				if k == vs[0]:
					v1 = vs[1]
				else:
					v1 = vs[0]
				new_clus[k].remove(v1)
				noadded.append(v1)

		return new_clus, noadded

	def get_inter_cand_pairs(self, old_clus, noadded, intra_pairs, sim_pairs, intra_ave_sim):
		""" low confidence synsets
		args:
			old_clus: {center: {'values':list(values), ave_sim:float(), sim_matrix}}}   #old clusters
			noadded: []                #values removed from initial cluster during before steps
			intra_pairs: {center:[]}   #candidate pairs used in previous steps
			sim_pairs:{[()]:float}   	
			intra_ave_sim: float       #mean similarity within a cluster
		returns:
			cand_pairs: {center:list}  #candidate center pairs that need input into LLMs
		"""
		all_centers = list(old_clus.keys())
		cand_pairs = {}
		# print('intra_ave_sim:', intra_ave_sim)
		for i in range(len(all_centers)-1):
			v1 = all_centers[i]
			cand_pairs[v1] = []
			temp_added = []
			for v2 in noadded:
				if v2 in old_clus[v1]['values']:
					continue
				else:
					temp_added.append(v2)

			for v2 in all_centers[i+1:]+temp_added:
				k = (v1.split('#****#')[0], v2.split('#****#')[0])
				if k in sim_pairs:
					sim = sim_pairs[k]
				else:
					k = (v2.split('#****#')[0], v1.split('#****#')[0])
					if k in sim_pairs:
						sim = sim_pairs[k]
					else:
						print(k, 'has no similarity')
						continue
				if sim > intra_ave_sim:
					cand_pairs[v1].append([v1, v2])
					print("candidate pairs", k, sim)

		return cand_pairs			
	
	def merge_clus_by_llm(self, old_clus, llm_res, noadded):
		"""merging clusters
		args:
			old_clus: {center:list}
			llm_res: {p:str}
			noadded: []   #values that removed from clusters
		returns:
			new_clus:{center:list}
		"""
		temp_cluster = []
		mappings = {}
		delete_idxs = set()
		temp_no_llm_res = []
		for k, info in llm_res.items():
			for pair, res in info.items():
				if 'no' not in res:
					continue
				vs = pair.split('--##--')
				v2 = vs[1]
				v1 = vs[0]

				temp_no_llm_res.append(vs)

		temp_old_clus = deepcopy(old_clus)
		for k, info in llm_res.items():
			for pair, res in info.items():
				if 'yes' not in res:
					continue

				vs = pair.split('--##--')
				v2 = vs[1]
				v1 = vs[0]

				# print(vs)
				if v2 in noadded:
					noadded.remove(v2)
				if v1 in noadded:
					noadded.remove(v1)

				if v1 in mappings:
					idx = mappings[v1]
					if v2 in mappings:
						if mappings[v1] != mappings[v2]:
							idx2 = mappings[v2]
							flag = True
							for v1_i in temp_cluster[idx]:
								for v2_i in temp_cluster[idx2]:
									if (v1_i, v2_i) in temp_no_llm_res:
										flag = False
										break
									elif (v2_i, v1_i) in temp_no_llm_res:
										flag = False
										break
								if not flag:
									break
							if flag:		
								# print(temp_cluster[idx], temp_cluster[idx2])
								delete_idxs.add(idx2)
								temp_cluster[idx].extend(temp_cluster[idx2])
								for x in temp_cluster[idx2]:
									if x in mappings:
										mappings[x] = idx
								# if v2 in temp_old_clus:
								# 	temp_cluster[idx].extend(temp_old_clus[v2])
								# 	del temp_old_clus[v2]
								# if v1 in temp_old_clus:
								# 	temp_cluster[idx].extend(temp_old_clus[v1])
								# 	del temp_old_clus[v1]

						else:
							continue
					else:
						mappings[v2] = idx
						temp_cluster[idx].append(v2)
						if v2 in temp_old_clus:
							temp_cluster[idx].extend(temp_old_clus[v2])
							del temp_old_clus[v2]
				else:
					if v2 in mappings:
						idx = mappings[v2]
						mappings[v1] = idx
						temp_cluster[idx].append(v1)
						if v1 in temp_old_clus:
							temp_cluster[idx].extend(temp_old_clus[v1])
							del temp_old_clus[v1]

					else:
						idx = len(temp_cluster)
						mappings[v1] = idx
						mappings[v2] = idx
						temp_cluster.append(vs)
						# print('new clusters:', vs, idx)
						if v2 in temp_old_clus:
							temp_cluster[idx].extend(temp_old_clus[v2])
							del temp_old_clus[v2]
						if v1 in temp_old_clus:
							temp_cluster[idx].extend(temp_old_clus[v1])
							del temp_old_clus[v1]

		# print('mappings:', mappings)
		new_clus = {}

		total = 0
		for idx, vs in enumerate(temp_cluster):
			if idx in delete_idxs:
				continue
			new_clus[vs[0]] = list(set(vs))
			total += len(new_clus[vs[0]])

		for k, vs in temp_old_clus.items():
			new_clus[k] = vs
			total += len(vs)

		for x in noadded:
			new_clus[x] = [x]	
			total += 1

		print('total：', total)	
		# print("new_clus:", new_clus) 
		return new_clus

	def old_merge_clus_by_llm(self, old_clus, llm_res, noadded):
		"""merge old clusters
		args:
			old_clus: {center:list}
			llm_res: {p:str}
			noadded: []   
		returns:
			new_clus:{center:list}
		"""
		new_clus = deepcopy(old_clus)
		mappings = {}
		delete_ks = set()
		for k, info in llm_res.items():
			for pair, res in info.items():
				if 'yes' not in res:
					continue

				vs = pair.split('--##--')
				# if vs[0] == vs[1]:
				# 	continue
				v2 = vs[1]
				if v2 in noadded:
					new_clus[k].append(v2)
					noadded.remove(v2)
					continue

				mappings[v2] = vs[0]

		for v2 in delete_ks:
			del new_clus[v2]

		turn = 0
		values = list(mappings.values())
		if values:
			counters = Counter(values)
			max_count = max(counters.values())
			max_elems = [elm for elm, counter in counters.items() if counter == max_count]

			for i in range(len(max_elems)-1, 0, -1):
				x = max_elems[i]
				for j in range(i-1, -1, -1):
					y = max_elems[j]
					if mappings.get(x, '') == y:
						 del max_elems[i]

			for x in max_elems:
				if x in mappings:
					mappings[mappings[x]] = x
					del mappings[x]		

			for x, y in mappings.items():
				turn = 0
				while y in mappings:
					y = mappings[y]
					turn += 1
					if turn > 10:
						print('--------excess the max turn:', len(mappings))
					break

			mappings[x] = y

			try:
				print("-------after mappings:", json.dumps(mappings, ensure_ascii=False))		
				for x, y in mappings.items():
					if y not in new_clus:
						new_clus[y] = new_clus.get(x, [x])
					else:
						new_clus[y].extend(new_clus.get(x, [x]))
						try:
							del new_clus[x]
						except:
							traceback.print_exc()

			except:
				traceback.print_exc()
				print("noadded:", noadded)
				print("mappings:", mappings)
				print("new_clus:", new_clus)

		for x in noadded:
			new_clus[x] = [x]	
		return new_clus

			
	def post4clus(self, old_clus, sim_pairs, llm_res):
		"""
		correct the old clustering results
		sim_pairs: similarity of all candidate pairs
		old_clus: list
		"""	
		temp_clus = {}
		all_ave_sims = []
		max_sim = max(list(sim_pairs.values()))
		min_sim = min(list(sim_pairs.values()))
		mid_sim = (max_sim+min_sim)/2

		print('the max, min, mid similarity:', max_sim, min_sim, mid_sim)
		intra_mins = []   #the minimum similarity within clusters
		intra_means =[]   #the average similarity within clusters
		for i, gather in enumerate(old_clus):   
			res = self.get_center(gather, sim_pairs)
			res['values'] = gather
			temp_clus[res['center']] = res
			if 'all_ave_sim' in res:
				intra_means.append(res['all_ave_sim'])
			if 'min_sim' in res:
				intra_mins.append(res['min_sim'])   

			if 'ave_sim' in res:
				all_ave_sims.append(res['ave_sim'])
				# self.ave_sims.append(res['ave_sim'])

		inter_sims = []
		centers = list(temp_clus.keys())
		for i, x in enumerate(centers):
			for y in centers[i+1:]:
				k = (x.split('#****#')[0],y.split('#****#')[0])
				if k in sim_pairs:
					inter_sims.append(sim_pairs[k])
				else:
					k = (y.split('#****#')[0], x.split('#****#')[0])
					if k in sim_pairs:
						inter_sims.append(sim_pairs[k])
					else:
						print(x, y, 'they have no similarity')
		inter_ave_sim = round(np.mean(inter_sims), 4)   
		self.inter_ave_sim[self.attr+'_'+self.con] = inter_ave_sim
		
		if inter_sims:
			inter_min_sim = np.min(inter_sims)
		# else:
		# 	inter_min_sim = inter_ave_sim
		self.inter_min_sims[self.attr+'_'+self.con] = inter_min_sim

		print("all_ave_sims: ", all_ave_sims)	
		intra_ave_sim = round(np.mean(all_ave_sims), 4)
		if np.isnan(intra_ave_sim):
			print('there is a error in similarity:', len(gather), intra_ave_sim)
		# intra_ave_sim = mid_sim if np.isnan(intra_ave_sim) else intra_ave_sim   #都是mid_sim
		intra_ave_sim = inter_ave_sim if np.isnan(intra_ave_sim) else intra_ave_sim

		self.intra_ave_sims[self.attr+'_'+self.con] = intra_ave_sim

		if intra_mins:
			intra_min = round(min(intra_mins), 4)
		else:
			if all_ave_sims:
				print('there is a error for intra_mins')
			intra_min = intra_ave_sim
		if intra_min < mid_sim:
			print('intra_min is less than mid similarity', intra_min, mid_sim)
		self.intra_min_sims[self.attr+'_'+self.con] = intra_min

		
		#return old_clus
		print('---------------------low confidence values:')
		print('---------------\nAverage of all intra-cluster similarity:', intra_ave_sim)
		intra_cand_pairs = self.get_intra_cand_pairs(temp_clus, intra_ave_sim)
		print('candidate pairs within clusters:', intra_cand_pairs)

		temp_llm_res = self.get_llm_anss(intra_cand_pairs, llm_res)

		new_clus, noadded = self.clean_clus_by_llm(temp_clus, temp_llm_res)
		print(json.dumps(temp_llm_res, ensure_ascii=False))

		print('---------------\nclusters after updated:', new_clus)
		print('values that removed from clusters:', noadded)

		print('----------------------low confidence synsets:')

		if 'intra' in self.post_res_file:
			print('intra_ave_sim is used as a threshod')
			inter_cand_pairs = self.get_inter_cand_pairs(temp_clus, noadded, intra_cand_pairs, sim_pairs, intra_ave_sim)
		
		elif 'inter' in self.post_res_file:
			print('Inter_min_sim is used as a threshod')
			inter_cand_pairs = self.get_inter_cand_pairs(temp_clus, noadded, intra_cand_pairs, sim_pairs, intra_min)  
		elif 'mid' in self.post_res_file:
			print('mid_sim is used as a threshod')
			inter_cand_pairs = self.get_inter_cand_pairs(temp_clus, noadded, intra_cand_pairs, sim_pairs, mid_sim)
		print('---------------\nall candidate pairs between clusters:', inter_cand_pairs)
		
		temp_llm_res = self.get_llm_anss(inter_cand_pairs, llm_res)
		new_clus1 = self.merge_clus_by_llm(new_clus, temp_llm_res, noadded)

		print('----------\nfinal clusters: new_clus1:', json.dumps(new_clus1, ensure_ascii=False))
		return new_clus1

	def convertformat(self, old_clus):
		"""convert data format from dict to list
		args:
			old_clus: {k:[]}
		returns:
			new_clus: [[],[]]
		"""
		new_clus = []
		for k, vs in old_clus.items():
			new_clus.append(vs)
		return new_clus

	def api(self):
		with open(self.cufen_res_file, 'r', encoding='utf-8') as fr:
			data = json.load(fr)

		if self.target_attrcons:
			keys = list(data.keys())
			if self.dataset == 'dbpedia':
				for k in keys:
					if k not in self.target_attrcons:
						del data[k] 
			elif self.dataset == 'cndbpedia':
				for k in keys:
					for v in list(data[k].keys()):
						if k+'_'+v not in self.target_attrcons:
							del data[k][v]
							if not data[k]:
								del data[k]

		if os.path.exists(self.llm_res_file):					
			with open(self.llm_res_file, 'r', encoding='utf-8') as fr:
				self.llm_data = json.load(fr)
		else:
			print('there is no answers from LLMs:', self.llm_res_file)
			self.llm_data = {}

		all_sim_pairs = {}	
		print('sim_res_file:', self.sim_res_file)
		with open(self.sim_res_file, 'r', encoding='utf-8') as fr:
			line = fr.readline()
			while line:
				if 'the attribute currently processed is' in line:
					attr = fr.readline().strip()
					line = fr.readline()
					con = fr.readline().strip()
					line = fr.readline()
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

		final_res = {}		
		for attr, info in data.items():
			print('-------the attribute currently processed is:', attr)
			self.attr = attr
			self.tosupple_pairs[attr] = {}
			final_res[attr] = {}
			if attr not in self.llm_data:
				print('there is no answers from LLMs', attr)
				self.llm_data[attr] = {}
			# if attr not in all_sim_pairs:
			# 	final_res[attr] = info
			# 	continue
			for	con, c_info in info.items():
				if self.tosupple_pairs_num > 30000:
					print('the calling number of LLM is too much, end early')
					break

				self.con = con
				self.tosupple_pairs[attr][con] = []
				if con not in self.llm_data[attr]:
					print('there is no answers from LLMs', attr, con)
					self.llm_data[attr][con] = {}
				print('--------the concept currently processed is: ', con)

				if self.dataset == 'dbpedia':
					self.prompt = self.new_prompts[attr]
				elif self.dataset == 'cndbpedia':
					self.prompt = self.new_prompts.get(attr+'_'+con, '')

				# if con not in all_sim_pairs[attr]:
				# 	final_res[attr][con] = c_info
				# 	continue
				# new_res = self.post4clus(c_info, all_sim_pairs[attr][con], llm_data[attr][con])

				new_res = self.post4clus(c_info, all_sim_pairs.get(attr, {}).get(con,{}), self.llm_data.get(attr, {}).get(con, {}))	
				# continue
				new_res = self.convertformat(new_res)
				final_res[attr][con] = new_res
			# 	break
			# break

		with open(self.post_res_file, 'w', encoding='utf8') as fw:
			json.dump(final_res, fw, ensure_ascii=False, indent=4)

		if self.added_llm_res:
			print('added_llm_res written into a file', self.llm_add_res_file)
			with open(self.llm_add_res_file, 'w', encoding='utf-8') as fw:
				json.dump(self.added_llm_res, fw, ensure_ascii=False, indent=4)	

			with open(self.temp_llm_res_file, 'w', encoding='utf-8') as fw:
				json.dump(self.llm_data, fw, ensure_ascii=False, indent=4)

		print('self.intra_ave_sims:')
		print(json.dumps(self.intra_ave_sims, ensure_ascii=False))		
		print('self.inter_ave_sim:')
		print(json.dumps(self.inter_ave_sim, ensure_ascii=False))	
		print('self.inter_min_sims:')
		print(json.dumps(self.inter_min_sims, ensure_ascii=False))		
		print('self.ave_sims:')
		print(json.dumps(self.ave_sims, ensure_ascii=False))
		print('self.intra_min_sims:')
		print(json.dumps(self.intra_min_sims, ensure_ascii=False))		
		print("new added queries:", self.tosupple_pairs_num)	   
		print("total number of queries that need put into LLMs:", self.needed_pairs)   
		print('the number of chars for new queries:', self.que_chars)
		print('the number of chars for new answers:', self.res_chars)

if __name__ == '__main__':
	postpre = Post()
	postpre.api()