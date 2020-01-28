from functools import reduce
import numpy as np
import json
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import os

def deleteDuplicate_v1(input_dict_lst):
	f = lambda x,y:x if y in x else x + [y]
	return reduce(f, [[], ] + input_dict_lst)

def get_context_pair(resp, l):
	label_weights = l['label_weights']
	valid_resp = {}
	for key in resp:
		valid_resp[key] = []
		for index, value in enumerate(resp[key]):
			if label_weights[index] == 1:
				valid_resp[key].append(value)

	answer = l['answer_tokens']
	position_tokens = l['tokens']
	label_position = [lpos-1 for index, lpos in enumerate(l['label_positions']) if label_weights[index]==1]
	
	score_label = []
	for index in range(len(valid_resp['pred_label'])):
		label = valid_resp['pred_label'][index]
		score = valid_resp['max_prob'][index]
		position = label_position[index]
		position_token = position_tokens[str(position)][1]
		if label == 1:
			score = 1 - score
		score_label.append({"score":score, "label":label, 
							"position_token":position_token,
						   "answer":answer})
	return score_label

def format_socre_matrix(result_lst, score_merge='mean'):
	answer_dict = {}
	candidate_dict = {}
	answer_index = 0
	pos_index = 0
	for item in result_lst:
		if item['answer'] not in answer_dict:
			answer_dict[item['answer']] = answer_index
			answer_index += 1
		if item['position_token'] not in candidate_dict:
			candidate_dict[item['position_token']] = pos_index
			pos_index += 1
			
	score_matrix = -np.ones((len(answer_dict), len(candidate_dict)))
	for item in result_lst:
		answer_pos = answer_dict[item['answer']]
		candidate_pos = candidate_dict[item['position_token']]
		score_matrix_score = score_matrix[answer_pos, candidate_pos]
		if score_matrix_score == -1:
			score_matrix[answer_pos, candidate_pos] = item['score']
		else:
			if score_merge == 'mean'
				score_matrix[answer_pos, candidate_pos] += item['score']
				score_matrix[answer_pos, candidate_pos] /= 2
			elif score_merge == 'max':
				if item['score'] > score_matrix[answer_pos, candidate_pos]:
					score_matrix[answer_pos, candidate_pos] = item['score']
	return score_matrix, answer_dict, candidate_dict

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

flags.DEFINE_string(
	"input_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"score_merge", None,
	"Input TF example files (can be a glob or comma separated).")

input_file = os.path.join(FLAGS.buckets, FLAGS.input_file)
output_file = os.path.join(FLAGS.buckets, FLAGS.output_file)
model_file = os.path.join(FLAGS.buckets, FLAGS.model_file)

from tensorflow.contrib import predictor
# model_dict = {
#     "model":'/data/xuht/albert.xht/nlpcc2019/open_data/model/1566283032'
# }

model_dict = {
	"model":model_file
}

chid_model = predictor.from_saved_model(model_dict['model'])

fwobj = tf.gfile.Open(output_file, "w")

cnt = 0
with tf.gfile.Open(input_file, "r") as f:
	for index, line in enumerate(f):
		content = json.loads(line.strip())
		total_resp = []
		for t in content:
			for l in t:
				tmp = {
					"input_ids":np.array([l['input_ids']]),
					'label_weights':np.array([l['label_weights']]),
					'label_positions':np.array([l['label_positions']]),
					'label_ids':np.array([l['label_ids']]),
					'segment_ids':np.array([l['segment_ids']]),
				}
				resp = chid_model(tmp)
				result = get_context_pair(resp, l)
				total_resp.extend(result)
		total_resp = deleteDuplicate_v1(total_resp)
		resp = format_socre_matrix(total_resp, score_merge=FLAGS.score_merge)
		row_ind, col_ind = linear_sum_assignment(resp[0])
		mapping_dict = dict(zip(col_ind, row_ind))
		
		candidte_dict = resp[-1]
		candidate_inverse_dict = {}
		for key in candidte_dict:
			candidate_inverse_dict[candidte_dict[key]] = key
		
		candidate_name_dict = {}
		for col in mapping_dict:
			col_name = candidate_inverse_dict[col]
			candidate_name_dict[col_name] = int(mapping_dict[col])
		cnt += len(candidate_name_dict)
		fwobj.write(json.dumps(candidate_name_dict, ensure_ascii=False)+"\n")
			
fwobj.close()
print('==total cnt==', cnt)

