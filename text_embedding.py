#coding:utf8
#get embeddings of text

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
		self.data_file = '/data/' + self.dataset + '_groundtruth.json'    

		#there are two pre-language models to get embedding, one is text-embedding-3-large provided by openai and the other is bert
		self.emb_tool = 'openai_large'
		self.emb_tool = 'bert'

		if 'openai' in self.emb_tool:
			openai.api_key = 'api_key*****'
			if 'large' in self.emb_tool:
				self.embedding_model = 'text-embedding-3-large'   #embedding size is 3072
				self.emd_file = self.dataset + '/embeddings_large.csv'
			else:
				print('unknown api')	
				
		elif self.emb_tool == 'bert':
			self.init_bert()				

			self.emd_file = self.dataset + '/bert_embeddings.csv'

	def read_file(self):
		with open(self.data_file, 'r', encoding='utf8') as fr:
			data = json.load(fr)
		values = set()
		lens = 0
		for attr, info in data.items():
			for c, vs in info['clusters']:
				for v in vs:
					values.add(v)
					lens += len(v)
			for v in info['single_vs']:
				values.add(v)
				lens += len(v)

		print('the number of values:', len(values))
		print('the number of tokens:', lens)
		return values

	def get_openapi(self, text):
		try:
			response = openai.Embedding.create(
				model = self.embedding_model,
				input = text)
			ans = response.data[0].embedding 
			return ans
		except:
			traceback.print_exc()
			print(text, 'error calling')
			return ''

	def init_bert(self):
		# initial BERT tokenizer
		if self.dataset == 'dbpedia':
			path = '/bert-base-uncased/'
		else:
			path = '/chinese-bert-wwm-ext/'
		self.tokenizer = BertTokenizer.from_pretrained(path)
		#initial bert model
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
		# sentence = "Hello world!"
		return [0]*100
		# tokenize
		new_sent = self.get_sequence(sentence)
		encoded_input = self.tokenizer(new_sent, return_tensors='pt')

		with torch.no_grad():
		    last_hidden_states = self.model(**encoded_input)
		    sentence_embedding = last_hidden_states.pooler_output

		sentence_embedding = sentence_embedding.numpy()
		 
		return sentence_embedding[0].tolist()

	def get_embedding(self, text='Female'):
		emb = 'error'
		# start_time = datetime.datetime.now()

		if self.emb_tool == 'openai':
			emb = self.get_openapi(text)
			if emb == '':
				emb = self.get_openapi(text)
		elif self.emb_tool == 'bert':
			emb = self.get_bert_emb(text)

		else:
			print('emb_tool is undefined ')
		
		# end_time = datetime.datetime.now()
		# print('cost times', end_time - start_time)
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
					print('index:', i)
		except:
			traceback.print_exc()
		finally:
			fw.close()

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
	docs = getemb.read_file()
	getemb.get_embs(docs)
	pairs = [["Private, Non-Profit", "State/Province"]]
	getemb.get_sims(pairs)

