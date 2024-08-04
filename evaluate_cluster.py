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
# from sklearn.metrics import 



class Eva_Clu:
	#对聚类后的结果进行评估
	"""
	macro-F1对每类的F1分数算算数平均值
	weighted-F1 考虑不同类别的重要性，每个类别的样本数量作为权重
	micro-F1全局的TP, FN和FP来计算F1分数
	"""
	def __init__(self):
		current_path, _ = os.path.split(os.path.realpath(__file__))
		# self.dir_ = 'E:\\research\\数据库清洗\\code'

		self.dataset = 'cndbpedia'
		self.dataset = 'dbpedia'
		if self.dataset == 'cndbpedia':
			self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_bigger.json'
			self.res_file =  self.dataset +'/'+self.dataset+'_cufen_res_cp_w_freq_0_louvain.json'
			#ari: 0.247866, v_measure: 0.810525, homos:0.870377, comps:0.797304, fmis: 0.173737,nmi:， acc:

			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_wp_tfidf_0_louvain.json'
			#ari: 0.203857, v_measure: 0.852298, homos:0.992373, comps:0.776073 fmis: 0.050377,nmi:， acc:

			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_cp_wo_freq_0_louvain.json'
			#ari: 0.224077, v_measure:0.818273, homos:0.898526, comps:0.788693, fmis:0.140825, nmi:， acc:

			# self.res_file = self.dataset + '/' +self.dataset+'_cufen_res_text_emb_small_0_louvain.json'
			#ari: 0.348544 , v_measure:0.804120, homos:0.823741, comps:0.831057, fmis:0.318479,nmi:， acc:

			# self.res_file = self.dataset + '/' +self.dataset+'_cufen_res_text_emb_large_0_louvain.json'  #0.7
			#ari: 0.382247 , v_measure:0.8301759, homos:0.862398, comps:0.838953, fmis:0.335530,nmi:0.830223, acc: 0.145611
			# self.res_file = self.dataset +'/' + 'cufen.json'   #利用字符级别的overlap
			#ari: 0.209279 , v_measure:0.546669, homos:0.452307, comps:0.926943, fmis:0.349747,

			# self.res_file = self.dataset + '/' +self.dataset+'_cufen_res_text_emb_large_0_louvain0.65.json'
			#ari:  0.365096, v_measure: 0.800684, homos:0.801681, comps:0.841713, fmis:0.351726, nmi:0.800684, acc:0.163146
			# self.res_file = self.dataset + '/' +self.dataset+'_cufen_res_text_emb_large_0_louvain0.6.json'
			#ari: 0.357353 , v_measure:0.770861, homos:0.740902, comps:0.859985, fmis:0.385453,nmi:0.770861, acc:0.181303
			
			suffix = '_post_qwen_max.json'
			suffix = '_post_gpt35.json'
			suffix = '_post_baichuan.json'
			suffix = '.json'
			# suffix = '_post_qwen_max.json'
			self.res_file =  self.dataset +'/'+self.dataset+'_cufen_res_text_emb_large_0_louvain0.67.json'
			self.res_file =  self.dataset +'/'+self.dataset+'_cufen_res_text_emb_large_cp_w_f7_0_louvain0.7.json'
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_large_tfidf5_0_louvain0.7.json'

			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res__dbscan_text_emb_large.json'
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res__kmeans_text_emb_large.json'

			self.res_file = self.dataset+'_cufen_res_text_emb_large_tfidf6_0_louvain0.7'+suffix   #self.dataset +'/'+

			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_wp_tfidf_0_louvain'+suffix
			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res__kmeans_text_emb_large'+suffix
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_roberta_0_louvain0.0'+suffix
			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_bert_0_louvain0.0'+suffix
			self.res_file = self.dataset +'/' +self.dataset +'_cufen_res_wp_tfidf_0_louvain0.0'+suffix
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res__kmeans_text_emb_large'+suffix
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_bert_seq_0_louvain0.0'+suffix
			self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_roberta_seq_l2s_0_louvain0.0'+suffix

			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_large_0_louvain0.0'+suffix
			# self.res_file = self.dataset +'/' +self.dataset +'_cufen_res_text_emb_roberta_wp_tfidf8_0_louvain0.0'+suffix

			# self.res_file = self.dataset +'/'+self.dataset+'_cufen_res_text_emb_large_1_louvain0.0.json'
			self.res_file = './final/'+self.dataset+'_cufen_res_wp_tfidf_0_louvain0.0_post_gpt35.json'
			self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_text_emb_large_0_louvain0.75'+suffix
			# self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_text_emb_large_tfidf8_0_louvain0.7'+suffix
			self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_text_emb_large_cp_w_f7_0_louvain0.7'+suffix
			# self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_wp_tfidf_0_louvain'+suffix
			# self.res_file = './final/cndbpedia_qwen_max_res_tfidf_0_-1_0.json'
			# print('比较的文件:', self.res_file)
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp8_0.json'
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp4_0_test.json'
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_0_test.json'  #最终答案，直接根据排序判断
			# self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_3_0_test.json' #最终答案，直接根据排序判断
			self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_topk.json' #最终答案，筛选topk
			self.res_file = './baselines/bert_cndbpedia_0.75_0.json'    #最终结果，用bert
			self.res_file = './cndbpedia_cufen_res_text_emb_large_wp_idf6_0_louvain0.0.json'    #最终结果，用louvain聚类
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp9_0_test.json'    #最终结果，用prompt2进行测试
			self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_15_intra.json'   #两个相似度都用intra来找低置信度点
			self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_15_mid.json'   #两个相似度都用intra来找低置信度点
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_15_inter.json'   #簇内用intra，簇间用inter_min

			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_-1_newp3_0_test.json'    #最终结果，用prompt3进行测试，每个属性最多调用100次
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_-1_0_test.json'    #最终结果，用prompt1进行测试，每个属性最多调用100次
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_100_newp9_0_test.json'    #最终结果，用prompt2进行测试，每个属性最多调用100次
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_100_newp10_0_test.json'    #最终结果，用prompt3+value进行测试，每个属性最多调用100次
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_100_newp4_0_test.json'    #最终结果，用prompt3+1shot进行测试，每个属性最多调用100次
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_100_newp8_0_test.json'    #最终结果，用prompt3+3shot进行测试，每个属性最多调用100次
			
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp3_0_test.json'    #最终结果，近义词，5次提前结束
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp3_0_test_0125.json'    #最终结果，近义词，5次提前结束, turbo-0125版本
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp8_0_test.json'    #最终结果，hac后用prompt3+3shot进行测试，turbo版本
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp3_0_test.json'    #最终结果，近义词，5次提前结束, turbo版本
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp11_0_test.json'    #最终结果，同一，5次提前结束, 随机3shot
			self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_200_0_test.json'    #最终结果，用prompt1进行测试，每个属性最多调用200次
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_p7_15_intra.json'    #最终结果，用hac后随机选prompt3+3shot进行测试
			# self.res_file = './final/cndbpedia_gpt35_res_large_cp_w_f6_0_5_newp7_0_test.json'    #最终结果，近义词 随机3shot，5次提前结束
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_gpt35_test.json'
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_baichuan_inter.json'
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75_post_qwen_max_inter.json'
			# self.res_file = './baselines/our_cndbpedia_text_emb_large_cp_w_f60.75.json'

			# self.res_file = self.dataset +'/qwen_max_res.json' 
			# self.res_file = self.dataset +'/qwen_max_res_large_5.json'
			# self.res_file = self.dataset +'/qwen_max_res_tfidf_3_newp.json'
			# self.res_file = self.dataset +'/qwen_turbo_res_large_3_newp1.json'
			# print('比较的文件:', self.res_file)
			self.target_attrcons = ['技术性质_车站', '国际濒危等级_学科', '设备种类_科技产品', '形象特征_人物', 
								'大坝类型_水电站', '成因类型_地貌', '所处时间_学科', '民居类型_地点', 
								'交建筑类型_建筑', '剧目类型_戏剧', '性别_人物', '行业类型_机构', '节目属性_娱乐', 
								'展馆类型_机构', '文章状态_小说']
			# self.target_attrcons = ['性别_人物', '交建筑类型_建筑', '形象特征_人物', '成因类型_地貌', '大坝类型_水电站']
			self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'.json'
			# self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_bigger.json'

		elif self.dataset == 'dbpedia':
			# ci_flag = '1'
			ci_flag = '0'
			#  suffix = '_post_gpt35.json'
			#  suffix = '_post_qwen_max.json'
			#  suffix = '_post_baichuan.json'
			# suffix = '.json'
			
			#self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_'+ci_flag+'.json'
			# self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_text_emb_large_tfidf4_'+ci_flag+'_louvain0.0'+suffix
			# self.res_file = self.dataset +'/cluster_res/'+self.dataset+'_cufen_res_text_emb_large_cp_w_f2_'+ci_flag+'_louvain0.0'+suffix
			# self.res_file = self.dataset + '/qwen_max_res_large_0_3.json'
			# # self.res_file = self.dataset + '/qwen_max_res_large_0_-1.json'
			# # self.res_file = self.dataset + '/qwen_max_res_large_0_3_jingjian.json'
			# self.res_file = self.dataset + '/qwen_max_res_tfidf_0_3_jingjian.json'
			# self.res_file = os.path.join(current_path, 'baselines/idftoken_dbpedia_0.75_0.json')
			# self.res_file = os.path.join(current_path, 'baselines/textsim_dbpedia_0.5.json')
			# self.res_file = os.path.join(current_path, 'baselines/our_cndbpedia_text_emb_large0.5.json')
			# # self.stand_file = os.path.join(current_path, 'dbpedia/stand_clusterres4dbpedia_0_bigger.json')
			self.target_attrcons = ['timeZone', 'broadcastArea', 'timezone1Dst', 'chairLabel', 'lakeType', 
								'areaBlank1Title', 'scoreboard', 'architectureType', 'batting', 
								'sworntype', 'stat1Header', 'link2Name', 'chrtitle']   #13个
								#'thirdRiderMoto2Country', 'membersLabel', 
			# self.target_attrcons = ['architectureType', 'chairLabel',  'timeZone']   #
			# # self.target_attrcons = ['batting']
		

			self.res_file = './baselines/idftoken_dbpedia_0.9_0.json'
			self.res_file = './baselines/textsim_dbpedia_0.9_0.json'
			# # self.res_file = './baselines/textsim_cndbpedia_0.9.json'
			# # self.res_file = './baselines/our_dbpedia_wp_tfidf0.75_0.json'
			# # self.res_file = './final/dbpedia_gpt35_res_tfidf_0_3_newp1_0.json'
			# # self.res_file = './final/dbpedia_gpt35_res_tfidf_0_-1_0.json'
			# self.res_file = './baselines/our_dbpedia_text_emb_large0.9_0.json'
			# self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0.json'
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.9_0_post_gpt35_p2_test.json'
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.9_0_post_llama_p2.json'
			# self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.9_0_post_qwen_max_p2.json'
			# self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.9_0_post_baichuan_p2.json'
			# self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.9_0.json'
			# self.res_file = './baselines/our_dbpedia_wp_tfidf0.5_0_post_qwen_max.json'
			# self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_0_bigger.json'
			# self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_5_newp1_0.json'
			# self.res_file = './final/dbpedia_gpt35_res_large_cp_w_0_5_newp4_0.json'
			self.res_file = './baselines/our_dbpedia_text_emb_large_wp_idf60.5_0.json'
			self.res_file = './baselines/bert_dbpedia_0.5_0.json'
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f6_0_5_newp2_0_test.json'  #最终答案，直接根据排序判断, 是f2，
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f6_0_3_newp2_0_test.json'  #最终答案，直接根据排序判断, 是f2，
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_gpt35_p2_topk.json' #最终答案，筛选topk
			self.res_file = './baselines/bert_dbpedia_0.9_0.json'    #最终结果，用bert
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_gpt35_p2_13_intra.json'   #两个相似度都用intra来找低置信度点
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_gpt35_p2_13_mid.json'   #两个相似度都用intra来找低置信度点
			# self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_gpt35_p2_13_inter.json'   #簇内用intra，簇间用inter_min
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_-1_newp0_0_test.json'    #最终结果，用prompt1进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_-1_0_test.json'    #最终结果，用prompt1进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_100_newp2_0_test.json'    #最终结果，用prompt3+3shot进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_100_newp4_0_test.json'    #最终结果，用prompt3+1shot进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_100_newp1_0_test.json'    #最终结果，用prompt3+value进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_5_newp5_0_test.json'    #最终结果，用prompt3+value+3shot进行测试，5次停止
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_-1_newp3_0_test.json'    #最终结果，用prompt3进行测试，每个属性最多调用100次
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_5_newp2_0_test.json'  #最终答案，直接根据排序判断, 是f2，
			self.res_file = './final/dbpedia_gpt35_res_large_cp_w_f2_0_5_newp3_0_test.json'  #最终答案，直接根据排序判断, 是f2，
			self.res_file = './baselines/our_dbpedia_text_emb_large_cp_w_f20.8_0_post_llama3_p2_15_intra.json'  #最终答案，直接根据排序判断, 是f2，
			self.stand_file = self.dataset + '/stand_clusterres4'+self.dataset+'_0.json'
		
		# self.target_attrcons = []
		print('stand_file:', self.stand_file)
		print('比较的文件:', self.res_file)


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

		# self.stand_data = {"小说状态":{"小说":[['完结', '已完结'], ['断更', '停更'], ['完本']], 
		# 						'名著':[['完结', '已完结'], ['断更', '停更'], ['完本']], 
		# 						'书籍':[['完结', '已完结'], ['断更', '停更'], ['完本','完本发布']]}}
		# self.res_data = {"小说状态":{"小说":[['完结', '已完结'], ['断更', '停更'], ['完本']],
		# 						'名著':[ ['断更', '停更'], ['完结', '已完结'], ['完本']], 
		# 						'书籍':[ ['断更', '停更'], ['完本发布', '完本', '完结', '已完结']]}}
		# 						#ari: 小说1.0， 名著1.0， 书籍0.4444
		# 						#homo:小说1.0， 名著1.0， 书籍0.5794
		# 						#comp:小说1.0， 名著1.0， 书籍1.0
		# 						#v_mea:小说1.0， 名著1.0， 书籍0.7337
		# 						#fmi:小说1.0， 名著1.0， 书籍0.65474
	def temp_convert_data(self, data_1, data_2):
		#预处理原始数据，将其输出为列表形式, data_1是标准答案，data_2是预测结果
		#data2: {k: [], k: [], k: []}
		#return : [0,0,1,2,3,4,3,2,1,1]
		v2labels = {}  #保存每个attr_value被聚类后的簇号， key: attr_value, value:[], 标准簇号，预测簇号

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
		#预处理原始数据，将其输出为列表形式, data_1是标准答案，data_2是预测结果
		#data_2: [[],[],[]]
		#return : [0,0,1,2,3,4,3,2,1,1]
		v2labels = {}  #保存每个attr_value被聚类后的簇号， key: attr_value, value:[], 标准簇号，预测簇号

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
			if v in ['股份上市（002236）']:
				continue
			if v not in v2labels:
				cluster_idx += 1
				v2labels[v] = [cluster_idx]

		# print("v2labels:", json.dumps(v2labels, ensure_ascii=False))
		# print('v2labels:', v2labels.get('Church with attached convent#****#附有修道院的教堂',[]))
		# print('v2labels:', v2labels.get('Church and convent#****#教堂和修道院',[]))

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
				if v in ['股份上市（002236）']:
					continue

				try:
					v2labels[v].append(cluster_idx)
					temp_v2labels[v] = cluster_idx
				except:
					traceback.print_exc()
					print("v2labels:", v2labels)
					print('v:', v)
					exit()
		# print('v2labels:', v2labels.get('Church and convent#****#教堂和修道院', []))
		# if 'Church and convent#****#教堂和修道院' in v2labels:
		# 	print(json.dumps(data_1, ensure_ascii=False))
		# 	print(json.dumps(data_2, ensure_ascii=False))

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
		Fowlkes-Mallows Index (FMI)定义为对精度(分组点对的准确性)和召回率(正确分组在一起的对的完整性)的几何平均值
		FMI = TP /sqrt((TP+FP)(TP+FN))
		[0-1], 其中0表示聚类结果与真实标签不相关，1表示完全相关。
		sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False)
		"""
		print('-----------FMI-------------')
		fmis = {}
		self.total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)

				assert(len(y_label) == len(y_test))
				fmi = round(fowlkes_mallows_score(y_label, y_test), 4)
				# print('属性:', attr, '概念:', con, 'fmi:', fmi)
				# fmis.append(fmi)
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
		同质性 Homogeneity 度量每个簇是否只包含单个类的成员。
		h = 1- H(C|K)/H(C)
		完整性 Completeness 度量给定类的所有成员是否被分配到同一个簇。
		c = 1- H(K|C)/H(K)
		V-measure是同质性和完备性的调和平均值，一个单一的分数来评估聚类性能。
		v = 2*hc/(h+c), [0-1], 越高越好
		sklearn.metrics.v_measure_score(labels_true, labels_pred, *, beta=1.0)

		"""
		print('-----------VMeasure-------------')
		homos = {}
		comps = {}
		v_measures = {}
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
				y_label, y_test = self.convert_data(c_info, pred_info)
				hom = round(homogeneity_score(y_label, y_test), 4)
				comp = round(completeness_score(y_label, y_test), 4)
				v_mea = round(v_measure_score(y_label, y_test), 4)
				# print('属性:', attr, '概念:', con, 'homo:', hom, 'comp:', comp, 'v_mea:', v_mea)
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
		两两比较来衡量聚类分配与真实类标签之间的相似性。
		计算簇分配和类标签之间的一致数与总数据点对数的比值: RI = (a+b)/Cn2
		ARI = (RI- E(RI))/(max(RI)-E(RI)), [-1,1], 越高越好
		sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
		"""
		#预处理原始数据成列表形式		
		print('-----------ARI-------------')
		aris = {}
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
				try:
					y_label, y_test = self.convert_data(c_info, pred_info)
				except:
					traceback.print_exc()
					print(attr, con)
					print(c_info)
					print(pred_info)
					exit()
				ari = round(adjusted_rand_score(y_label, y_test), 4)
				# print('属性:', attr, '概念:', con, 'ari:', ari)
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

	def DBI(self):
		"""
		衡量每个聚类与其最相似的聚类之间的平均相似度，其中相似度定义为聚类内距离与聚类间距离之比
		DBI = 
		[0, ∞], 越小越好。
		"""
		pass 

	def NMI(self):
		"""
		NMI，归一化互信息，度量两个聚类的相近程度， {0-1]
		NMI(Y,C) = 2*I(Y;C)/(H_Y+H_C)
		I(Y;C) = H_Y - H(Y|C)  互信息，一个随机变量中包含的关于另一个随机变量的信息量，表示两个集合的相关性
		H_Y = -SUM(P_i*logP_i)  真实数据标签的交叉熵
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
		聚类的准确率，匈牙利算法
		ACC = S
		sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
		"""
		print('-----------ACC-------------')
		accs = {}
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
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
		right = {}   #分子
		total = {}   #分母
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
		# 		pred_info = self.res_data[attr][con]
				# print('--------\n',attr)
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
					count.append(np.bincount(labels_tmp).max()) #用于统计每个数据出现的次数
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
		#将簇转化成候选对
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
		macro记录预测正确的簇的比例
		micro，记录预测正确的元素的比例
		pairwise，元素对正确的比例

		计算对的pre，rec， f1
		tp: true positive 正预测正, tn:true negative 负预测负, 
		fp: false positive 负预测正,  fn: false negative 正预测负
		pre = tp/(tp+fp)
		rec = tp/(tp+fn)
		weight按候选值个数加权算平均
		"""
		pres = {}
		recs = {}
		f1s = {}
		tps = 0
		fps = 0 
		fns = 0
		N = 0 #所有属性值的个数
		C_N = 0 #预测结果的簇的个数
		G_N = 0 #标准答案的簇的个数
		G_N_S =0  #同义词组，大于2的簇的个数
		pre_c = 0  #被标准答案包含的簇的个数
		rec_g = 0  #被预测结果包含的簇的个数
		max_c = 0   #和标准答案重叠最多的个数
		max_g = 0   #和预测结果重叠最多的个数

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
				#统计pairwise P, R 的相关指标
				stand_res = self.convert_to_pairs(list(c_info['clusters'].values()))
				pred_res = self.convert_to_pairs(pred_info)
				tp = len(stand_res & pred_res)
				fp = len(pred_res)-tp
				fn = len(stand_res)-tp
				tps += tp
				fps += fp
				fns += fn	

				#统计macro P, R和micro P，R的相关指标
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
							if not(c - g):   #g包含c
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
							if not(g - c): # c包含g
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
		"""macro记录预测正确的簇的比例
		micro，记录预测正确的元素的比例
		pairwise，元素对正确的比例
		"""
		pres = {}
		recs = {}
		f1s = {}
		tps = 0
		fps = 0 
		fns = 0
		N = 0 #所有属性值的个数
		C_N = 0 #预测结果的簇的个数
		G_N = 0 #标准答案的簇的个数
		G_N_S =0  #同义词组，大于2的簇的个数
		pre_c = 0  #被标准答案包含的簇的个数
		rec_g = 0  #被预测结果包含的簇的个数
		max_c = 0   #和标准答案重叠最多的个数
		max_g = 0   #和预测结果重叠最多的个数
		same_c = 0   #和标准答案一摸一样的簇的个数

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

				print('------------------\n')
				print(attr)

				# print('c_info:', c_info)
				# print('pred_info:', pred_info)
				#统计pairwise P, R 的相关指标
				stand_res = self.convert_to_pairs(list(c_info['clusters'].values()))
				pred_res = self.convert_to_pairs(pred_info)
				tp = len(stand_res & pred_res)
				fp = len(pred_res)-tp
				fn = len(stand_res)-tp
				tps += tp
				fps += fp
				fns += fn	

				#统计macro P, R和micro P，R的相关指标
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
						if (not set(c-g)) and (not set(g-c)):   #两个簇完全一样
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
							if not(c - g):   #g包含c
								pre_c += 1
								flag_p = False
						# 		if g-c:
						# 			print('###\ng包含c pre_c:', g-c, c, g)


						max_overl = max(max_overl,  len(c & g))
						
					max_c += max_overl
					
					# if (not same) and len(diff) < len(tag_g):
					# 	print('###\ng包含c pre_c:', diff, tag_g, c)
				
				for g in stand_res:
					flag_r = True
					max_overl = 0 
					same = set()
					tag_c = set()
					for c in new_pred_info:
						max_overl = max(max_overl, len(c & g))
						if flag_r:
							if not(g - c): # c包含g
								rec_g += 1
								flag_r = False
								# if c-g:
								# 	print('*****\nc包含g rec_g:', c-g,  c, g)
						
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
						print('*****\nc包含g rec_g:', tag_c-g, '\n', g, '\n', 'tag_c:', tag_c)

		
		print('N:', N, 'C_N:', C_N, 'G_N:', G_N, 'G_N_S:', G_N_S)
		print('max_c:', max_c, 'max_g:', max_g, 'pre_c:', pre_c, 'rec_g:', rec_g)	
		if tps+fps ==0:
			print('有问题，', tps, fps)	
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

	def F1(self):
		"""计算值对的pre，rec， f1
		tp: true positive 正预测正, tn:true negative 负预测负, 
		fp: false positive 负预测正,  fn: false negative 正预测负
		pre = tp/(tp+fp)
		rec = tp/(tp+fn)
		macro各算各的，最后平均
		micro所有类别的一起算
		weight按候选值个数加权算平均
		"""
		pres = {}
		recs = {}
		f1s = {}
		tps = 0
		fps = 0 
		fns = 0
		for attr, info in self.res_data.items():
			for con, pred_info in info.items():
				c_info = self.stand_data[attr][con]
		# for attr, info in self.stand_data.items():
		# 	for con, c_info in info.items():
				stand_res = self.convert_to_pairs(list(c_info['clusters'].values()))
				pred_res = self.convert_to_pairs(pred_info)
				tp = len(stand_res & pred_res)
				fp = len(pred_res)-tp
				fn = len(stand_res)-tp
				tps += tp
				fps += fp
				fns += fn		
				# print(attr, con)
				# # print(stand_res, pred_res)
				# print(tp, fp, fn)	
				k = attr+'_'+con
				if tp+fp == 0:
					pres[k] = 1.0
				else:	
					pres[k] = round(tp*1.0/(tp+fp), 4)
				if tp+fn == 0:
					recs[k] = 1.0
				else:
					recs[k] = round(tp*1.0/(tp+fn), 4)

				# print(pres[k], recs[k])
				if pres[k]+recs[k] == 0.0:
					f1s[k] = round(2*pres[k]*recs[k]/1.0,4)

				else:	
					f1s[k] = round(2*pres[k]*recs[k]/(pres[k]+recs[k]),4)
				
		macro_pre = round(np.mean(np.array(list(pres.values()))), 4)
		macro_rec = round(np.mean(np.array(list(recs.values()))), 4)
		macro_f1 = round(np.mean(np.array(list(f1s.values()))), 4)

		micro_pre = round(tps*1.0/(tps+fps), 4)
		micro_rec = round(tps*1.0/(tps+fns), 4)
		micro_f1 = round(micro_rec*micro_pre*2/(micro_pre+micro_rec), 4)
		
		weight_avg_pre = 0.0
		weight_avg_rec = 0.0
		weight_avg_f1 = 0.0
		fenmu = np.sum(list(self.total.values()))
		for k, v in self.total.items():
			weight_avg_pre += pres[k] * v / fenmu
			weight_avg_rec += recs[k] * v / fenmu
			weight_avg_f1 += f1s[k] * v / fenmu

		return {'macro_rec':macro_rec, 'macro_pre':macro_pre, 'macro_f1':macro_f1,
				'micro_rec':micro_rec, 'micro_pre':micro_pre, 'micro_f1':micro_f1,
				'weight_avg_pre':round(weight_avg_pre, 4), 'weight_avg_rec':round(weight_avg_rec, 4),
				'weight_avg_f1': round(weight_avg_f1, 4)}

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
	# with open('C:\\Users\\Admin\\Desktop\\result.txt', 'w', encoding='utf-8') as fw:
	# 	fw.write(s+'\n')
	exit()

	aris_res = eval_cluster.ARI()
	v_meas_res = eval_cluster.VMeasure()
	fmis_res = eval_cluster.FMI()
	purs_res = eval_cluster.PURITY()
	nmis_res = eval_cluster.NMI()
	accs_res = eval_cluster.ACC()
	f1_res = eval_cluster.F1()

	for x in [aris_res, v_meas_res, fmis_res, purs_res, nmis_res, accs_res]:
		print(str(x['macro'])+', '+str(x['weight_avg']), end =', ')

	for x in ['macro_pre', 'macro_rec', 'macro_f1', 'micro_pre', 'micro_rec', 'micro_f1', 'weight_avg_pre', 'weight_avg_rec', 'weight_avg_f1']:
		print(f1_res[x], end=', ')


	print()
	file = 'metric.csv'
	with open(file, 'w', encoding='utf-8') as fw:
		for k, value in aris_res['aris'].items():
			fw.write(k+','+str(round(value, 4))+','+str(round(v_meas_res['v_measures'][k], 4))+','+str(round(fmis_res['fmis'][k], 4))+','+str(round(purs_res['puritys'][k], 4))+','+str(round(nmis_res['nmis'][k], 4))+','+str(round(accs_res['accs'][k], 4))+'\n')
