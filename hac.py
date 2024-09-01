#coding:utf8
#using hac based on the similarity
import os
import json
from idftoken import *
import traceback
import optparse

class HAC:
	"""three criteria:
	1. single linking
	2. complete linking
	3. average linking
	"""
	def __init__(self, thre=0.5, print_flag=True):
		self.method = 'single'
		self.thre = thre   #threshold for ending
		self.print_flag = print_flag

	def hcluster(self, docs, sim_dict):
		#docs: all values that need to be clustered
		#sim_dict: similarity between all value pairs

		sorted_sims = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)

		(v1, v2), max_sim = sorted_sims[0]
		if self.print_flag:
			print('max similarity:', sorted_sims[0])
		if max_sim < self.thre:
			res_clu = [[x] for x in docs]
			return res_clu

		temp_clu = {}
		c_idx = 0 #current number of clusters
		new_clu = [v1, v2]
		docs.remove(v1)
		docs.remove(v2)
		# temp_clu['#c'+str(c_idx)] = new_clu
		if self.print_flag:
			print('new added cluster: new_clu：', '#c'+str(c_idx), new_clu)
			print('docs after updated:', docs, len(docs))

		temp_sim_dict = copy.deepcopy(sim_dict)
		del_cs = set()   #deleted cluster idx
		while True:
			if self.print_flag:
				print('--------------------------')
				print('cluster idx, c_idx:', c_idx)
			for k in list(temp_sim_dict.keys()):
				if (k[0] in new_clu) or (k[1] in new_clu) or ( k[0] in del_cs) or (k[1] in del_cs):
					del temp_sim_dict[k]

			# if self.print_flag:
			# 	print('temp_sim_dict:', temp_sim_dict.keys())
			
			for x in docs:
				temp_dist = 0.0
				for v in new_clu:
					if self.method == 'single':
						temp_dist = max(temp_dist, sim_dict.get((x, v), 0.0), sim_dict.get((v, x), 0.0))
				temp_sim_dict[('#c'+str(c_idx), x)] = temp_dist
			# if self.print_flag:
			# 	print('temp_sim_dict:', temp_sim_dict.keys())

			for name, vs in temp_clu.items():
				temp_dist = 0.0
				for v1 in vs:
					for v2 in new_clu:
						if self.method == 'single':
							temp_dist = max(temp_dist, sim_dict.get((v1, v2), 0.0), sim_dict.get((v2, v1), 0.0))	
				temp_sim_dict[('#c'+str(c_idx), name)] = temp_dist

			# if self.print_flag:
			# 	print('temp_sim_dict:', temp_sim_dict.keys())

			temp_clu['#c'+str(c_idx)] = new_clu
			if self.print_flag:
				print('new added cluster, temp_clu：', temp_clu)
				# print('temp_sim_dict:', temp_sim_dict)

			if not temp_sim_dict:
				break

			sorted_sims = sorted(temp_sim_dict.items(), key=lambda x:x[1], reverse=True)	
			(v1, v2), max_sim = sorted_sims[0]
			if self.print_flag:
				print('max similarity:', sorted_sims[0])

			if max_sim < self.thre:
				break
			
			new_clu = []
			if v1.startswith('#c'):
				new_clu.extend(temp_clu[v1])
				del_cs.add(v1)
				del temp_clu[v1]
			else:
				new_clu.append(v1)
				try:
					docs.remove(v1)
				except:
					traceback.print_exc()
					print(v1)
					print(docs)

			if v2.startswith('#c'):
				new_clu.extend(temp_clu[v2])
				del_cs.add(v2)
				del temp_clu[v2]
			else:
				new_clu.append(v2)
				docs.remove(v2)

			c_idx += 1
			
			if self.print_flag:
				print('temp_clu:', temp_clu)
				print('new_clu:', new_clu)
				print('docs updated:', docs, len(docs))

		res_clu = [v for k, v in temp_clu.items()] +  [[x] for x in docs]

		if self.print_flag:
			print('remain docs:', docs)
			print('added clu:', temp_clu )
			print('############')
			print('res_clu:', res_clu)
		return res_clu


class Ours:
	def __init__(self, thre=0.9):
		self.lang = 'EN'
		# self.lang = 'CH'
		current_path, _ = os.path.split(os.path.realpath(__file__))
		self.thre = thre
		self.tag = 'text_emb_large'
		self.tag = 'wp_tfidf'
		self.tag = 'text_emb_large_tfidf4'
		self.tag = 'text_emb_large_cp_w_f5'
		self.tag = 'text_emb_large_wp_idf6'
		self.tag = 'bert'
		if self.lang == 'EN':
			self.simfile = os.path.join(current_path, '../dbpedia/sim_pairs_file_'+self.tag+'_0.json')
			self.standfile = os.path.join(current_path, '../dbpedia', 'stand_clusterres4dbpedia_0.json')
			self.res_file = os.path.join(current_path, self.tag+'_dbpedia_'+str(self.thre)+'_0.json')

		else:
			if 'emb' in self.tag and 'cp' in self.tag:
				self.simfile = os.path.join(current_path, '../cndbpedia/sim_pairs_file_'+self.tag+'_0.json') 	
			else:	
				self.simfile = os.path.join(current_path, '../cndbpedia/sim_pairs_file_'+self.tag+'_0.json') 	
			self.standfile = os.path.join(current_path, '../cndbpedia', 'stand_clusterres4cndbpedia.json')
			self.res_file = os.path.join(current_path, self.tag+'_cndbpedia_'+str(self.thre)+'_0.json')

		self.hac = HAC(self.thre, False)	

	def get_cluster(self):
		docs_dict = {}
		mapping_dict = {}
		cluster_res = {}				
		with open(self.standfile, 'r', encoding='utf-8') as fr:
			data = json.load(fr)
			for attr, info in data.items():
				if attr not in docs_dict:
					docs_dict[attr] = {}
					mapping_dict[attr] = {}
					cluster_res[attr] = {}
				for con, c_info in info.items():
					docs = set()
					clusters = c_info['clusters']
					for k, vs in clusters.items():
						docs |= set(vs)

					docs |= set(c_info['noadded_vs'])
					docs = list(docs)
					docs_dict[attr][con] = docs
					mapping_dict[attr][con] = {}
					for x in docs:
						new_x = x.split('#****#')[0]
						# mapping_dict[attr][con][new_x] = x
						mapping_dict[attr][con][x] = x

		with open(self.simfile, 'r', encoding='utf-8') as fr:
			line = fr.readline()
			while line:
				if "the attribute currently processed is:" in line:
					attr = fr.readline().strip().strip()
					line = fr.readline()
					con = fr.readline().strip()
					sims = json.loads(fr.readline())
					
					# if attr not in ['lakeType', 'endmo']:
					# 	line = fr.readline()
					# 	continue

					print('----------------\nthe attribute currently processed is::', attr)
					print('the concept currently processed is:', con)

					sim_dict = {}
					for k, v in sims.items():
						try:
							ks = k.split('--##--')
							new_k = (mapping_dict[attr][con][ks[0]], mapping_dict[attr][con][ks[1]])

							sim_dict[new_k] = v
						except:
							traceback.print_exc()
							print(k, v)

					cluster_res[attr][con] = self.hac.hcluster(docs_dict[attr][con], sim_dict)	

				line = fr.readline()

		with open(self.res_file, 'w', encoding='utf-8') as fw:
			json.dump(cluster_res, fw, ensure_ascii=False, indent=4)

if __name__ == '__main__':
	mdopt = optparse.OptionParser()
	mdopt.add_option('-t', '--thre', dest='threshold', type='float', default=0.5)
	options, args = mdopt.parse_args()
	thre = options.threshold

	us = Ours(thre)	
	us.get_cluster()		