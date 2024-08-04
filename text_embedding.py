#coding:utf8
#获取指定文本的embedding结果

import openai
import os
import traceback
import datetime 
import csv
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertModel, BertTokenizer
import torch

class Emb:
	def __init__(self):

		self.dataset = 'dbpedia'	
		self.data_file = self.dataset + '/stand_cleanres4' + self.dataset + '.json'    #属性值数:10294,token数:71502

		# self.dataset = 'dbpedia'
		# self.data_file = self.dataset +'/stand_cleanres4' + self.dataset +'.json'      #属性值数: 24159, token数: 642369

		self.emd_file = self.dataset + '/bert_embeddings.csv'
		self.emd_file = self.dataset + '/test.csv'   #max_length= setence_len+2
		self.emd_file = self.dataset + '/bert_seq_embeddings_l2s.csv'  #从所有token到1个token开始的序列
		self.emd_file = self.dataset + '/robert_seq_embeddings_l2s.csv'  #从所有token到1个token开始的序列
		self.emd_file = self.dataset + '/robert_seq_embeddings_s2l.csv'  #从所有token到1个token开始的序列
		self.emd_file = self.dataset + '/embeddings_large_0_1.csv'

		self.emb_tool = 'openai'
		self.emb_tool = 'bert'
		# # self.emb_tool = ''

		if self.emb_tool == 'openai':
			openai.api_key = '*****'

		# 	# self.embedding_model = 'text-embedding-ada-002'    #1536
		# 	self.embedding_model = 'text-embedding-3-small'   #1536
			self.embedding_model = 'text-embedding-3-large'   #3072 ,
		# 	# miracl 指标 adav2:31.4%, small:44.0%, large:54.9%
		# 	# MTEB 指标 adav2: 61.0%, small: 62.3%, large:64.9%
		# 	# embedding_encoding ='cl100k_base'
		# 	# self.client = OpenAI()
		elif self.emb_tool == 'bert':
			self.init_bert()				

	def compare(self):
		self.data_file = self.dataset + '/stand_clusterres4' + self.dataset + '.json'
		with open(self.data_file, 'r', encoding='utf8') as fr:
			data = json.load(fr)

		all_names = set()
		for k, info in data.items():
			for c, vs in info['cluster'].items():
				for v in vs:
					v = v.split('#****#')[0]
					all_names.add(v)
			for v in info['noadded_vs']:	
				v = v.split('#****#')[0]
				all_names.add(v)

		with open(self.emd_file, 'r', encoding='utf8') as fr:
			reader = csv.reader(fr)
			for row in reader:
				# print(row)
				name, emb = row
				all_names.discard(name)	
		
		print(len(all_names))
		print(list(all_names))

	def read_cn_file(self):
		with open(self.data_file, 'r', encoding='utf8') as fr:
			data = json.load(fr)
		values = set()
		for attr, info in data.items():
			for c, c_info in info.items():
				values |= set(c_info)

		print('属性值数：', len(values))
		#统计token数
		lens = sum(len(x) for x in values)
		print('token数：', lens)
		return values

	def read_file(self):
		with open(self.data_file, 'r', encoding='utf8') as fr:
			data = json.load(fr)
		values = set()
		lens = 0
		for attr, vs in data.items():
			for v in vs:
				v = v.split('#****#')[0]
				values.add(v)
				lens += len(v)

		print('属性值数：', len(values))
		#统计token数
		# lens = sum(len(x) for x in values)
		print('token数：', lens)
		return values

	def get_openapi(self, text):
		try:
			response = openai.Embedding.create(
				model = self.embedding_model,
				input = text)
			# response = self.client.embeddings.create(input = [text], model=self.embedding_model)
			ans = response.data[0].embedding 
			# print(ans)
			# print(type(ans))
			# # print(ans.shape())
			return ans
		except:
			traceback.print_exc()
			print(text, '出错')
			return ''

	def init_bert(self):
		# 加载BERT tokenizer
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		path = '/mnt/heying/resources/chinese_bert_wwm_ext_L-12_H-768_A-12'
		path = '/mnt/heying/resources/chinese-bert-wwm-ext/'
		if 'robert' in self.emd_file:
			path = '/mnt/heying/resources/chinese-roberta-wwm-ext/'
		elif 'bert' in self.emd_file:
			path = '/mnt/heying/resources/chinese-bert-wwm-ext/'
		self.tokenizer = BertTokenizer.from_pretrained(path)
		 
		# 加载BERT模型
		self.model = BertModel.from_pretrained(path)
	
	def get_sequence(self, sentence):
		sub_seqs = defaultdict(list)
		new_sents = []
		start_idx = 0
		for i in range(len(sentence)):
			if not self.is_ch_or_en(sentence[i]):
				if start_idx != i:
					new_sents.append(sentence[start_idx:i])
				start_idx = i+1
		if start_idx != len(sentence):
			new_sents.append(sentence[start_idx:])
		for x in new_sents:
			for i in range(len(x)):
				# print('i=', i)
				for j in range(1, len(x)-i+1):
					# print('j=', j, x[i:i+j])
					sub_seqs[j].append(list(x[i:i+j]))
		# print(sub_seqs)	
		if not sub_seqs:
			sub_seqs[len(sentence)] = [[sentence]]
		try:		
			max_len = max(list(sub_seqs.keys()))
		except:
			traceback.print_exc()
			print(sentence)
			exit()

		new_seq = []	
		if 'l2s' in self.emd_file:
			if sentence not in new_sents:
				new_seq += list(sentence)
			# for i in range(max_len+1):
			for i in range(max_len, -1, -1):
				for x in sub_seqs[i]:
					if len(new_seq + x) > 509:
						break
					new_seq.extend(['[SEP]']+x)
		
		elif 's2l' in self.emd_file:
			new_seq += list(sentence)
			for i in range(max_len):
				for x in sub_seqs[i]:
					if len(new_seq+x) > 509:
						break
					new_seq = x+['[SEP]']+new_seq	
		# print(len(new_seq))
		new_seq = ''.join(new_seq)
		return new_seq
				

	def is_ch_or_en(self, char):
		#ch
		if '\u4e00' <= char <= '\u9fff':
			return True		
		if '\u3400' <= char <= '\u4dbf':
			return True
		#en
		if '\u0020' <= char <= '\u007F':
			return True		

		return False

	def get_bert_emb(self, sentence):	 
		# 输入句子
		# sentence = "Hello world!"
		 
		# 对句子进行tokenize
		new_sent = self.get_sequence(sentence)
		# print(new_sent)
		encoded_input = self.tokenizer(new_sent, return_tensors='pt')
		# print(encoded_input)

		# if len(encoded_input) > 512:
		# 	encoded_input = encoded_input[:511]+['[SEP]']
		# print(sentence, encoded_input)

		# 获取嵌入
		with torch.no_grad():
		    last_hidden_states = self.model(**encoded_input)
		    sentence_embedding = last_hidden_states.pooler_output
		 
		# 转换为numpy数组
		sentence_embedding = sentence_embedding.numpy()
		 
		# print(sentence_embedding)
		return sentence_embedding[0].tolist()

	def get_embedding(self, text='女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】'):
		emb = '出错'
		# start_time = datetime.datetime.now()

		if self.emb_tool == 'openai':
			emb = self.get_openapi(text)
			if emb == '':
				emb = self.get_openapi(text)
		elif self.emb_tool == 'bert':
			emb = self.get_bert_emb(text)

		else:
			print('未考虑的编码方式')
		
		# end_time = datetime.datetime.now()
		# print('耗费时间：', end_time - start_time)
		return emb

	def get_embs(self, docs):
		try:
			fw = open(self.emd_file, 'w', newline='', encoding='utf8')
			writer = csv.writer(fw)
			for i, x in enumerate(docs):
				res = self.get_embedding(x)
				writer.writerow([x, res])
				# break
				if i % 3000 == 1:
					print('处理到', i)
		except:
			traceback.print_exc()
		finally:
			fw.close()

	def read_embs(self):
		self.all_embs = {}
		with open(self.emd_file, 'r', encoding='utf8') as fr:
			reader = csv.reader(fr)
			for row in reader:
				# print(row)
				name, emb = row
				emb = json.loads(emb)
				self.all_embs[name] = emb
		print('一共的属性值数:', len(self.all_embs.keys()))		

	def get_sims(self, pairs):
		for (v1, v2) in pairs:
			if hasattr(self, 'all_embs') :
				emb1 = self.all_embs[v1]
				emb2 = self.all_embs[v2]
			else:
				emb1 = self.get_bert_emb(v1)
				emb2 = self.get_bert_emb(v2)

			sim = round(cosine_similarity([emb1], [emb2])[0][0], 4)

			print(v1, v2, sim)


if __name__ == '__main__':
	getemb = Emb()
	# # getemb.compare()
	# # exit()

	# docs = ['Telephone code. 02995', 'scrapped preserved', 'Boys Mixed', 'Head Coach Pavel Tresnak manager = Oiva Tapio', 'Regionsubdivision_name1 = Inner Mongolia', 'left. -handed', 'Telephone/ STD code = 0326', 'taluka subdivision_name2 = pallipatu', 'Male Female', 'Country =India', 'English Telugu', 'Print paperback Digital eBook', 'County, Medicare, Medicaid', 'District subdivision_name2 = Purbi Singhbhum', 'Country subdivision_name =', 'Telephone code = +91', 'Dialing code. = +34', 'CLST utc_offset_DST = -3']		
	# # docs = ["男", 'TJ', '女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】']
	# # for x in docs:
	# # 	# new_seq = getemb.get_sequence(x)
	# # 	getemb.get_bert_emb(x)
	# # 	# print(new_seq)
	# # exit()
	# # docs = ['女', '完结']
	# # docs = getemb.read_file()
	# # docs = getemb.read_cn_file()
	# # docs = ['舞蹈公开赛', '现已停更', '男性阿斯泰坦', '印度']
	# # docs = ['Private, Non-Profit', 'On display at the Wishram depot in Wishram, Washington', 'State/Province', 'Village in', 'Active as civilian tug Noelani', 'Neo-Classical and Georgian Palladian', 'Northwest-Southeast', 'Biggest Village', 'Mainly Hmong,Regional Chinese, Thai, Vietnamese, Lao, French, English, Burmese', 'Novel, stage show, television show', 'color:white; background:#1E59AE;', 'good', 'local heritage', 'Suburban, gated community', 'Jawi', 'Silky, vitreous to dull', 'West-east-south', 'TeluguTamil', 'Ceremonial Chief', 'Colorless, greenish, greyish yellow, white, pink', 'Vitreous, pearly, greasy', 'department', 'Numbered Regions', 'Neighbourhoods', 'Speaker of the Batasan', 'Bikol', 'Liturgical language', 'Diesel, Diesel-electric', 'freely accessible', 'Dark brown to dark greenish-black', 'Editor at Large', 'Mixed-sex education, coed', 'Dark to light red-brown', 'color:white; background:#095339;', 'Kingdom of England', 'Black 90%, Mulatto, 8 % Carib-Amerindian 2%', 'Bulgarian', 'Extant suborders and superfamilies', 'United Kingdom, mainland Europe', 'Chairman of the Supreme Soviet', 'Home base', "Sheriff's Deputy", 'Royal emblem', 'Features Editor Commissioning Editor', 'Scout.com Football Recruiting: Aubun', 'Cream, red, black, gold, apricot, brown, white or a combination', 'Bura-Pabir', 'Tambon', 'City of Ottawa']
	# # docs = ["Private, Non-Profit", "On display at the Wishram depot in Wishram, Washington", "State/Province", "Village in"]
	# getemb.get_embs(docs)
	# print(getemb.emd_file)
	# getemb.read_embs()                                                                                                                                                                                                                                                                                                            
	# pairs = [['北周', '中国东周'], ['中国中国', '中中国'], ['中国x', '中国1'], 
	# 		['清朝（中国）', '中国（晚清）'], ['中华人人共和国', '中或人民共和国'], ['中国（北宋）', '宋（今中国）']]
	# getemb.get_sims(pairs)
	# getemb.emd_file = getemb.dataset + '/embeddings_large_0_1.csv'
	# print(getemb.emd_file)
	# getemb.get_sims(pairs)

	#合并两个embedding文件
	all_embs = defaultdict(list)
	target_embs = ['Head Coach Pavel Tresnak manager = Oiva Tapio', 'Regionsubdivision_name1 = Inner Mongolia', 'left. -handed','Telephone/ STD code = 0326']
	for i in range(len(target_embs)):
		temp = target_embs[i].lower()
		emb1 = getemb.get_bert_emb(temp)
		emb2 = getemb.get_bert_emb(target_embs[i])
		sim = round(cosine_similarity([emb1], [emb2])[0][0], 4)   #更换大小写相似度为1.0
		print(temp, target_embs[i], sim)

# 	#当前处理的文件embeddings_large
# 	with open('./dbpedia/embeddings_large_temp.csv', 'r', encoding='utf8') as fr:
# 		reader = csv.reader(fr)
# 		for row in reader:
# 			name, emb = row
# 			temp_name = name.replace(' ', '').lower()
# 			if temp_name not in target_embs:
# 				continue
# 			print(name)
# 			try:
# 				emb = json.loads(emb)
# 			except:
# 				traceback.print_exc()
# 				print('emb：',row)
# 				continue
# 				# exit()
# 			all_embs[temp_name].append(emb)


# 	#当前处理的文件embeddings_large_0_1
# 	print('当前处理的文件embeddings_large_0_1')
# 	with open('./dbpedia/embeddings_large_0_1.csv', 'r', encoding='utf8') as fr:
# 		reader = csv.reader(fr)
# 		for row in reader:
# 			# print(row)
# 			name, emb = row
# 			temp_name = name.replace(' ', '').lower()
# 			if temp_name not in target_embs:
# 				continue
# 			print(name)
# 			try:
# 				emb = json.loads(emb)
# 			except:
# 				traceback.print_exc()
# 				print('emb：',row)
# 			all_embs[temp_name].append(emb)

# for x, embs in all_embs.items():
# 	print(x)
# 	for i in range(1, len(embs)):
# 		sim = round(cosine_similarity([embs[0]], [embs[i]])[0][0], 4)			
# 		print(sim)