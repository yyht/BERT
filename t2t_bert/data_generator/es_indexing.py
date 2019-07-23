try:
	import ujson as json
except:
	import json

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.helpers import bulk
from collections import OrderedDict
import os, sys

class ESSearch(object):
	def __init__(self, config=None):

		if isinstance(config, dict) or isinstance(config, OrderedDict):
			self.config = config
		elif isinstance(config, str):
			try:
				self.config = json.load(open(config, "r"))
			except:
				self.config = {}

		self.username = self.config.get("username", "data_security_es_45")
		self.password = self.config.get("password", "Nb6121ca7ffe3")
		es_url = self.config.get("es_url", ['http://zsearch.alipay.com:9999'])

		if isinstance(es_url, list):
			self.es_url = es_url
		else:
			self.es_url = [es_url]

		self.es = Elasticsearch(self.es_url, http_auth=(self.username, self.password))

	def create(self, index):
		self.es.indices.create(index=index)

	def delete(self, index):
		self.es.indices.delete(index=index)

	def index_batch_doc(self, doc_index, doc_type, data_chunk, batch_size):
		# try:
		# 	self.es.indices.create(index=doc_index)
		# 	print("=========succeeded in creating doc index============", doc_index)
		# except:
		# 	self.delete(doc_index)
		# 	print("=========succeeded in deleting doc index============", doc_index)
		actions = []
		batch_index = 0
		batch_num = int(float(len(data_chunk))/float(batch_size))
		end_index = 0
		for index in range(batch_num):
			start_index = index * batch_size
			end_index = (index+1) * batch_size
			for row in data_chunk[start_index:end_index]:
				actions.append({
						"_op_type":"index",
						"_index":doc_index,
						"_type":doc_type,
						"_source":row
				})			
			bulk(self.es, actions)
			actions = []
		if end_index < len(data_chunk):
			actions = []
			for row in data_chunk[end_index:]:
				actions.append({
						"_op_type":"index",
						"_index":doc_index,
						"_type":doc_type,
						"_source":row
					}) 
			bulk(self.es, actions)
		# print("==succeeded in indexing es for===", doc_index, doc_type)

	def index_one_batch_doc(self, doc_index, doc_type, data_chunk, batch_size):
		actions = []
		for row in data_chunk:
			actions.append({
					"_op_type":"index",
					"_index":doc_index,
					"_type":doc_type,
					"_source":row
			})			
		bulk(self.es, actions)

	def index_single_doc(self, index, doc_type, body, idx):
		self.es.index(index=index,
					doc_type=doc_type,
					body=body,
					id=idx)

	def get_all_doc(self, index, doc_type, search_body):
		results = helpers.scan(
								client=self.es,
								query=search_body,
								scroll="60s",
								index=index,
								doc_type=doc_type,
								timeout="60s")
		results_list = []
		for item in results:
			results_list.append(item["_source"])
		return results_list

	def search_doc(self, index, search_body, threshold=0.5, size=10):
		results = self.es.search(index=index,
								body=search_body,
								size=size,
								timeout="60s")
		outputs = []
		if len(results["hits"]["hits"]) >= 1:
			max_score = results["hits"]["max_score"]+1e-10
			for idx, item in enumerate(results["hits"]["hits"]):
				normalized_score = item["_score"]/max_score
				if normalized_score >= threshold:
					outputs.append({"score":item["_score"], "source":item['_source']})
		return outputs
