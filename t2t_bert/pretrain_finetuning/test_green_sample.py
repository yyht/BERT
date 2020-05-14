#-*- coding: utf-8 -*-

import requests
import numpy as np
import json
import concurrent.futures
import codecs

with codecs.open('./test_1.txt', 'r', 'utf-8') as frobj:
	input1 = frobj.read().strip()

with codecs.open('./candidate_1.txt', 'r', 'utf-8') as frobj:
	candidate1 = frobj.read().strip()

with codecs.open('./test_2.txt', 'r', 'utf-8') as frobj:
	input1 = frobj.read().strip()

with codecs.open('./candidate_2.txt', 'r', 'utf-8') as frobj:
	candidate1 = frobj.read().strip()

post_data_1 = {
		"data":{
				"query":input1,
				"candidate":[candidate1]
		}
}

def create_http_session(config):
		session = requests.Session()
		a = requests.adapters.HTTPAdapter(max_retries=config.get("max_retries", 3),
								  pool_connections=config.get("pool_connections", 100),
								  pool_maxsize=config.get("pool_maxsize", 100))
		session.mount('http://', a)
		return session
session = create_http_session({})


def infer_data():
	headers = {}
	headers["Authorization"] = "ZWE5Y2FmNTgxMjA2NzdmOTJlOTEyMTllNmFkMTI4MDg4ZDk5OGMzYQ=="
	response = requests.post("http://11.31.153.212:58756/api/predict/pi_text_similarity_match_v1_bj_90ebb4d6",
				  data=json.dumps(input_data))

	results = (response.content)
	return results

resp = infer(post_data_1)
print(resp)
