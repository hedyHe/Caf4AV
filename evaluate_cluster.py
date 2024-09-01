#coding:utf8

import json
import os
import traceback
import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score


class Eva_Clu:
	#evaluating the final clustering results
	"""
	"""
	def __init__(self):
		current_path, _ = os.path.split(os.path.realpath(__file__))

		self.dataset = 'cndbpedia'
		self.dataset = 'dbpedia'
		if self.dataset == 'cndbpedia':
			self.stand_file = '/stand_clusterres4'+self.dataset+'_bigger.json'
			self.res_file =  self.dataset +'/'+self.dataset+'_cufen_res_cp_w_freq_0_louvain.json'
			
			self.target_attrcons = ['技术性质_车站', '国际濒危等级_学科', '设备种类_科技产品', '形象特征_人物', 
								'大坝类型_水电站', '成因类型_地貌', '所处时间_学科', '民居类型_地点', 
								'交建筑类型_建筑', '剧目类型_戏剧', '性别_人物', '行业类型_机构', '节目属性_娱乐', 
								'展馆类型_机构', '文章状态_小说']
			self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'.json'

		elif self.dataset == 'dbpedia':
			ci_flag = '0'
			self.target_attrcons = ['timeZone', 'broadcastArea', 'timezone1Dst', 'chairLabel', 'lakeType', 
								'areaBlank1Title', 'scoreboard', 'architectureType', 'batting', 
								'sworntype', 'stat1Header', 'link2Name', 'chrtitle']   #13
		
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_llama3_p2_15_intra.json' 
			self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_0.json'
		
		print('standard file:', self.stand_file)
		print('predict file:', self.res_file)


		with open(self.stand_file, 'r', encoding='utf-8') as fr:
			self.stand_data = json.load(fr)

		with open(self.res_file, 'r', encoding='utf-8') as fr:
			self.res_data = json.load(fr)
			if self.target_attrcons:
				keys = list(self.res_data.keys())
				for k in keys:
					if self.dataset == 'dbpedia':
						if k not in self.target_attrcons:
							del self.res_data[k]
					else:
						info = self.res_data[k]
						cons = list(info.keys())
						for con in cons:
							if k+'_'+con not in self.target_attrcons:
								del self.res_data[k][con]
								if not self.res_data[k]:
									del self.res_data[k] 

	def temp_convert_data(self, data_1, data_2):
		#covert original data into a list, data_1 is standard data, and data_2 is predict results
		#data2: {k: [], k: [], k: []}
		#return : [0,0,1,2,3,4,3,2,1,1]
		v2labels = {}  #key: attr_value, value:[stand_id, pred_id], 

		clusters = data_1['clusters']
		noadded = data_1['noadded_vs']
		cluster_idx = 0
		for i, (tag, cluster) in enumerate(clusters.items()):
			inter = set(cluster) & set(v2labels.keys())
			if inter:
				for v in inter:
					cluster_idx = v2labels[v][0]
					break
			else:
				cluster_idx = i
			other = set(cluster) - inter
			for v in other:
				v2labels[v] = [cluster_idx]

		for v in noadded:
			cluster_idx += 1
			v2labels[v] = [cluster_idx]

		# print("v2labels:", json.dumps(v2labels, ensure_ascii=False))

		temp_v2labels = {}
		for i, (k, cluster) in enumerate(data_2.items()):
			inter = set(cluster) & set(temp_v2labels.keys())

			if inter:
				for v in inter:
					cluster_idx = temp_v2labels[v]
					break
			else:
				cluster_idx = i 
			other = set(cluster) - inter
			for v in other:
				try:
					v2labels[v].append(cluster_idx)
					temp_v2labels[v] = cluster_idx
				except:
					traceback.print_exc()
					# print(v2labels, v)

		y_label = []
		y_test = []
		for v, idxs in v2labels.items():
			l, t = idxs
			y_label.append(l)
			y_test.append(t)

		return y_label, y_test

	def convert_data(self, data_1, data_2):
		#convert the original data into a One-dimensional list 
		#data_1: standard answers [[],[],[]]
		#data_2: predict answers [[],[],[]]
		#return : [0,0,1,2,3,4,3,2,1,1]
		v2labels = {}  #Save the cluster idx for each attr value,  key: attr_value, value:[standard idx, predict idx], 

		clusters = data_1['clusters']
		noadded = data_1['noadded_vs']
		cluster_idx = 0
		clusters = sorted(clusters.items(), key= lambda x:len(x[1]), reverse=True)
		# for i, (tag, cluster) in enumerate(clusters.items()):
		for i, (tag, cluster) in enumerate(clusters):
			inter = set(cluster) & set(v2labels.keys())
			if inter:
				for v in inter:
					cluster_idx = v2labels[v][0]
					break
			else:
				cluster_idx = i
			other = set(cluster) - inter
			for v in other:
				v2labels[v] = [cluster_idx]

		for v in noadded:
			if v not in v2labels:
				cluster_idx += 1
				v2labels[v] = [cluster_idx]


		temp_v2labels = {}
		for i, cluster in enumerate(data_2):
			inter = set(cluster) & set(temp_v2labels.keys())

			if inter:
				for v in inter:
					cluster_idx = temp_v2labels[v]
					break
			else:
				cluster_idx = i 
			other = set(cluster) - inter
			for v in other:
				try:
					v2labels[v].append(cluster_idx)
					temp_v2labels[v] = cluster_idx
				except:
					traceback.print_exc()
					print("v2labels:", v2labels)
					print('v:', v)
					exit()

		y_label = []
		y_test = []
		for v, idxs in v2labels.items():
			try:
				l, t = idxs
				y_label.append(l)
				y_test.append(t)
			except:
				traceback.print_exc()
				print(v, idxs)
				exit()

		return y_label, y_test

	def FMI(self):
		"""
		sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False)
		"""
		print('-----------FMI-------------')
		fmis = {}
		self.total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)

				assert(len(y_label) == len(y_test))
				fmi = round(fowlkes_mallows_score(y_label, y_test), 4)
[]				# fmis.append(fmi)
				fmis[attr+'_'+con] = fmi
				self.total[attr+'_'+con] = len(y_label)
		average = np.mean(np.array(list(fmis.values())))
		# print('fmis:', json.dumps(fmis, ensure_ascii=False))	
		print('average:', average)	
		weight_avg = 0.0
		fenmu = np.sum(list(self.total.values()))
		for x, v in fmis.items():
			weight_avg += v * self.total[x] / fenmu
		print('weight_avg:', weight_avg)
		return {'macro':round(average, 4), 'weight_avg':round(weight_avg,4), 'fmis':fmis} 
			
	def VMeasure(self):
		"""
		sklearn.metrics.v_measure_score(labels_true, labels_pred, *, beta=1.0)

		"""
		print('-----------VMeasure-------------')
		homos = {}
		comps = {}
		v_measures = {}
		total = {}   #denominator
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)
				hom = round(homogeneity_score(y_label, y_test), 4)
				comp = round(completeness_score(y_label, y_test), 4)
				v_mea = round(v_measure_score(y_label, y_test), 4)
				# homos.append(hom)
				# comps.append(comp)
				# v_measures.append(v_mea)
				v_measures[attr+'_'+con] = v_mea
				homos[attr+'_'+con] = hom
				comps[attr+'_'+con] = comp
				total[attr+'_'+con] = len(y_label)
		average = np.mean(np.array(list(v_measures.values())))
		# print('v_measures:', json.dumps(v_measures, ensure_ascii=False))		
		print('average:', average)	

		weight_avg = 0.0
		fenmu = np.sum(list(total.values()))
		for x, v in v_measures.items():
			weight_avg += v * total[x] / fenmu
		print('weight_avg:', weight_avg)

		# print('homos:', json.dumps(homos, ensure_ascii=False))
		# average = np.mean(np.array(list(homos.values())))
		# print('homos average:', average)	
		# print('comps:', json.dumps(comps, ensure_ascii=False))
		# average = np.mean(np.array(list(comps.values())))
		# print('comps average:', average)
		return {'macro':round(average, 4), 'weight_avg':round(weight_avg,4), 'v_measures':v_measures} 

	def ARI(self):
		"""
		sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
		"""
		print('-----------ARI-------------')
		aris = {}
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
				try:
					y_label, y_test = self.convert_data(c_info, pred_info)
				except:
					traceback.print_exc()
					print(attr, con)
					print(c_info)
					print(pred_info)
					exit()
				ari = round(adjusted_rand_score(y_label, y_test), 4)
				# print('attribute:', attr, 'concept:', con, 'ari:', ari)
				# aris.append(ari)
				aris[attr+'_'+con] = ari
				total[attr+'_'+con] = len(y_label)

		# print('ari:', json.dumps(aris, ensure_ascii=False))		
		average = np.mean(np.array(list(aris.values())))
		print('average:', average)	
		weight_avg = 0.0
		fenmu = np.sum(list(total.values()))
		for x, v in aris.items():
			weight_avg += v * total[x] / fenmu
		print('weight_avg:', weight_avg)
		return {'macro':round(average, 4), 'weight_avg':round(weight_avg,4), 'aris':aris} 

	def NMI(self):
		"""
		sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
		"""
		print('-----------NMI-------------')
		nmis = {}
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)
				nmi = round(normalized_mutual_info_score(y_label, y_test), 4)
				nmis[attr+'_'+con] = nmi
				total[attr+'_'+con] = len(y_label)

		# print('nmi:', json.dumps(nmis, ensure_ascii=False))		
		average = np.mean(np.array(list(nmis.values())))
		print('average:', average)
		weight_avg = 0.0
		fenmu = np.sum(list(total.values()))
		for x, v in nmis.items():
			weight_avg += v * total[x]/ fenmu
		print('weight_avg:', weight_avg)	
		return {'macro':round(average, 4), 'weight_avg':round(weight_avg,4), 'nmis':nmis} 

	def ACC(self):
		"""
		sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
		"""
		print('-----------ACC-------------')
		accs = {}
		total = {}   
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)
				acc = round(accuracy_score(y_label, y_test), 4)
				accs[attr+'_'+con] = acc
				total[attr+'_'+con] = len(y_label)

		# print('accs:', json.dumps(accs, ensure_ascii=False))		
		average = np.mean(np.array(list(accs.values())))
		print('average:', average)	
		weight_avg = 0.0
		fenmu = np.sum(list(total.values()))
		for x, v in accs.items():
			weight_avg += v * total[x]/ fenmu
		print('weight_avg:', weight_avg)
		return {'macro': round(average, 4), 'weight_avg':round(weight_avg,4), 'accs': accs}

	def PURITY(self):
		print('-----------PURITY-------------')
		puritys = {}
		right = {}   #molecule of the equation
		total = {}   #denominator of the equation
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]

				y_label, y_test = self.convert_data(c_info, pred_info)

				clusters = np.unique(y_label)
				labels_true = np.reshape(y_label, (-1, 1))
				labels_pred = np.reshape(y_test, (-1, 1))
				count = []
				num = 0
				for c in clusters:
					idx = np.where(labels_pred==c)[0]
					labels_tmp = labels_true[idx, :].reshape(-1)
					if len(labels_tmp) ==0 :
						# print(idx, labels_true, labels_pred)
						# break
						num += 1
						continue
					count.append(np.bincount(labels_tmp).max()) 
				puritys[attr+'_'+con] = np.sum(count) / labels_true.shape[0]
				right[attr+'_'+con] = np.sum(count)
				total[attr+'_'+con] = labels_true.shape[0]

				# print(attr, con, num)
		# print('puritys:', json.dumps(puritys, ensure_ascii=False))
		average = np.mean(np.array(list(puritys.values())))
		print('average:', average)
		togather = np.sum(list(right.values())) /np.sum(list(total.values()))
		print('togather:', togather)
		weight_avg = 0.0
		fenmu = np.sum(list(total.values()))
		for x, v in puritys.items():
			weight_avg += v * total[x] / fenmu
		print('weight_avg:', weight_avg)
		return {'macro': round(average, 4), 'micro': round(togather, 4), 'weight_avg':round(weight_avg,4), 'puritys': puritys}

	def convert_to_pairs(self, data):
		#data: [[,,,],[],[]]
		#return : ['---##--','---##--','---##--']
		res = set()
		for vs in data:
			for i in range(len(vs)-1):
				for j in range(i+1, len(vs)):
					res.add(vs[i]+'---##---'+vs[j])
					res.add(vs[j]+'---##---'+vs[i])
		return res	


	def new_F1_by_w(self):
		"""
		"""
		pres = {}
		recs = {}
		f1s = {}
		tps = 0
		fps = 0 
		fns = 0
		N = 0      #the number of all values
		C_N = 0    #the number of predicte clusters
		G_N = 0    #the number of standard clusters
		G_N_S =0   #the number of clusters who have more than 2 values
		pre_c = 0  #the number of predict clusters that be included in a standard cluster
		rec_g = 0  #the number of standard clusters that be included in a predict cluster
		max_c = 0   #the max number of values in predicte clusters that overlaps the standard clusters
		max_g = 0   #the max number of values in standard clusters that overlaps the predicte clusters

		def convert(data, G_N, G_N_S):
			clusters = data['clusters']
			noadded = data['noadded_vs']
			all_values = set(noadded)
			new_clusters = []
			for tag, cluster in clusters.items():
				new_clusters.append(set(cluster))
				all_values |= new_clusters[-1]
				G_N += 1
				G_N_S += 1

			for x in noadded:
				new_clusters.append(set([x]))
				G_N += 1

			return new_clusters, G_N, G_N_S, len(all_values)

		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]

				print('c_info:', c_info)
				print('pred_info:', pred_info)
				stand_res = self.convert_to_pairs(list(c_info['clusters'].values()))
				pred_res = self.convert_to_pairs(pred_info)
				tp = len(stand_res & pred_res)
				fp = len(pred_res)-tp
				fn = len(stand_res)-tp
				tps += tp
				fps += fp
				fns += fn	

				stand_res, G_N, G_N_S, num = convert(c_info, G_N, G_N_S)
				N += num
				C_N += len(pred_info)
				new_pred_info = []
				for c in pred_info:
					c = set(c)
					new_pred_info.append(c)
					max_overl = 0
					flag_p = True
					for g in stand_res:
						if flag_p:
							if not(c - g):   #g includes c
								print('pre_c:', c, g)
								pre_c += 1
								flag_p = False
						max_overl = max(max_overl,  len(c & g))
					max_c += max_overl
				
				for g in stand_res:
					flag_r = True
					max_overl = 0 
					for c in new_pred_info:
						max_overl = max(max_overl, len(c & g))
						if flag_r:
							if not(g - c): # c includes g
								print('rec_g:', c, g)
								rec_g += 1
								flag_r = False
					max_g += max_overl
		
		print('N:', N, 'C_N:', C_N, 'G_N:', G_N, 'G_N_S:', G_N_S)
		print('max_c:', max_c, 'max_g:', max_g, 'pre_c:', pre_c, 'rec_g:', rec_g)		
		pairw_pre = round(tps*1.0/(tps+fps), 4)
		pair_rec = round(tps*1.0/(tps+fns), 4)
		pair_f1 = round(pairw_pre*pair_rec*2/(pairw_pre+pair_rec), 4)
		
		macro_pre = round(pre_c*1.0/C_N, 4) 
		macro_rec = round(rec_g*1.0/G_N, 4)
		macro_f1 = round(macro_pre*macro_rec*2/(macro_rec+macro_pre), 4)
		micro_pre = round(max_c*1.0/N, 4)
		micro_rec = round(max_g*1.0/N, 4)
		micro_f1 = round(micro_pre*micro_rec*2/(micro_pre+micro_rec), 4)

		print('-----------macro-------------')
		print('pre:', macro_pre)
		print('rec:', macro_rec)
		print('f1:', macro_f1)
		print('-----------micro-------------')
		print('pre:', micro_pre)
		print('rec:', micro_rec)
		print('f1:', micro_f1)
		print('-----------pair-------------')
		print('pre:', pairw_pre)
		print('rec:', pair_rec)
		print('f1:', pair_f1)

		return {'macro_rec':macro_rec, 'macro_pre':macro_pre, 'macro_f1':macro_f1,
				'micro_rec':micro_rec, 'micro_pre':micro_pre, 'micro_f1':micro_f1,
				'pairw_pre':pairw_pre, 'pair_rec':pair_rec,
				'pair_f1': pair_f1}


	def new_F1(self):

		pres = {}
		recs = {}
		f1s = {}
		tps = 0
		fps = 0 
		fns = 0
		N = 0      #the number of all values
		C_N = 0    #the number of predicte clusters
		G_N = 0    #the number of standard clusters
		G_N_S =0   #the number of clusters who have more than 2 values
		pre_c = 0  #the number of predict clusters that be included in a standard cluster
		rec_g = 0  #the number of standard clusters that be included in a predict cluster
		max_c = 0   #the max number of values in predicte clusters that overlaps the standard clusters
		max_g = 0   #the max number of values in standard clusters that overlaps the predicte clusters
		same_c = 0   #the number of predict clusters that are same with standard clusters

		def convert(data, G_N, G_N_S):
			clusters = data['clusters']
			noadded = data['single_vs']
			all_values = set(noadded)
			new_clusters = []
			for tag, cluster in clusters.items():
				new_clusters.append(set(cluster))
				all_values |= new_clusters[-1]
				G_N += 1
				G_N_S += 1

			for x in noadded:
				new_clusters.append(set([x]))
				G_N += 1

			return new_clusters, G_N, G_N_S, len(all_values)

		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]

				print('------------------\n')
				print(attr)

				# print('c_info:', c_info)
				# print('pred_info:', pred_info)
				stand_res = self.convert_to_pairs(list(c_info['clusters'].values()))
				pred_res = self.convert_to_pairs(pred_info)
				tp = len(stand_res & pred_res)
				fp = len(pred_res)-tp
				fn = len(stand_res)-tp
				tps += tp
				fps += fp
				fns += fn	

				stand_res, G_N, G_N_S, num = convert(c_info, G_N, G_N_S)
				N += num
				C_N += len(pred_info)
				new_pred_info = []
				for c in pred_info:
					c = set(c)
					new_pred_info.append(c)
					max_overl = 0
					flag_p = True
					diff = set()
					tag_g = set()
					same = False
					for g in stand_res:
						if (not set(c-g)) and (not set(g-c)):  
							same_c += 1
							same = True

						elif not (c-g):
							if len(diff) == 0:
								diff = g-c
								tag_g = g
							elif len(g-c) < len(diff):
								diff = g - c
								tag_g = g	

						if flag_p:
							if not(c - g):   #g includes c
								pre_c += 1
								flag_p = False

						max_overl = max(max_overl,  len(c & g))
						
					max_c += max_overl
					
				
				for g in stand_res:
					flag_r = True
					max_overl = 0 
					same = set()
					tag_c = set()
					for c in new_pred_info:
						max_overl = max(max_overl, len(c & g))
						if flag_r:
							if not(g - c): # c includes  g
								rec_g += 1
								flag_r = False
						
						if max_overl != len(g):
							# if not (g-c):
							if len(c&g) > len(same):
								same = c&g
								tag_c = c
						else:
							if len(c&g) == len(g):
								same = g
								tag_c = c

					max_g += max_overl
					
					if len(same) < len(g):
						print('*****\nc includes g rec_g:', tag_c-g, '\n', g, '\n', 'tag_c:', tag_c)

		
		print('N:', N, 'C_N:', C_N, 'G_N:', G_N, 'G_N_S:', G_N_S)
		print('max_c:', max_c, 'max_g:', max_g, 'pre_c:', pre_c, 'rec_g:', rec_g)	
		if tps+fps ==0:
			print('there are some errors in this process', tps, fps)	
		pairw_pre = round(tps*1.0/max(tps+fps, 1), 4)
		pair_rec = round(tps*1.0/(tps+fns), 4)
		pair_f1 = round(pairw_pre*pair_rec*2/(pairw_pre+pair_rec), 4)
		
		exact_pre = round(same_c*1.0/C_N, 4)
		exact_rec = round(same_c*1.0/G_N, 4)
		exact_f1 = round(exact_pre*exact_rec*2/(exact_pre+exact_rec), 4)
		macro_pre = round(pre_c*1.0/C_N, 4) 
		macro_rec = round(rec_g*1.0/G_N, 4)
		macro_f1 = round(macro_pre*macro_rec*2/(macro_rec+macro_pre), 4)
		micro_pre = round(max_c*1.0/N, 4)
		micro_rec = round(max_g*1.0/N, 4)
		micro_f1 = round(micro_pre*micro_rec*2/(micro_pre+micro_rec), 4)

		print('-----------macro-------------')
		print('pre:', macro_pre)
		print('rec:', macro_rec)
		print('f1:', macro_f1)
		print('-----------micro-------------')
		print('pre:', micro_pre)
		print('rec:', micro_rec)
		print('f1:', micro_f1)
		print('-----------exact-------------')
		print('pre:', exact_pre)
		print('rec:', exact_rec)
		print('f1:', exact_f1)
		print('-----------pair-------------')
		print('pre:', pairw_pre)
		print('rec:', pair_rec)
		print('f1:', pair_f1)

		return {'macro_rec':macro_rec, 'macro_pre':macro_pre, 'macro_f1':macro_f1,
				'micro_rec':micro_rec, 'micro_pre':micro_pre, 'micro_f1':micro_f1,
				'pairw_pre':pairw_pre, 'pair_rec':pair_rec, 'pair_f1': pair_f1,
				'exact_pre':exact_pre, 'exact_rec': exact_rec, 'exact_f1': exact_f1}


if __name__ == '__main__':
	eval_cluster = Eva_Clu()

	purs_res = eval_cluster.PURITY()
	nmis_res = eval_cluster.NMI()
	f1_res = eval_cluster.new_F1()

	print(f1_res['macro_pre'], f1_res['macro_rec'], f1_res['macro_f1'], f1_res['micro_pre'], f1_res['micro_rec'], f1_res['micro_f1'], end=' ')
	print(f1_res['pairw_pre'], f1_res['pair_rec'], f1_res['pair_f1'], purs_res['macro'], purs_res['micro'], purs_res['weight_avg'], end=' ')
	print(nmis_res['macro'], nmis_res['weight_avg'], end=' ')
	print(f1_res.get('exact_pre', 0.0), f1_res.get('exact_rec', 0.0), f1_res.get('exact_f1', 0.0))

	s = ''
	for k in ["macro_pre", "macro_rec", "macro_f1", "micro_pre", "micro_rec", "micro_f1", "pairw_pre", "pair_rec", "pair_f1"]:
		s += str(f1_res[k]) +','
	s += str(purs_res['macro']) +',' +str(purs_res['micro']) +',' +str(purs_res['weight_avg']) +',' +str(nmis_res['macro']) +',' +str(nmis_res['weight_avg'])
	s += ','+ str(f1_res.get('exact_pre', 0.0)) +','+ str(f1_res.get('exact_rec', 0.0))+','+ str(f1_res.get('exact_f1', 0.0))
	print(s)
	exit()
