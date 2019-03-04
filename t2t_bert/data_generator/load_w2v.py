import tensorflow as tf
import numpy as np
from collections import OrderedDict

def load_pretrained_w2v(w2v_path):
	with tf.gfile.Open(w2v_path) as frobj:
		lines = []
		for line in frobj:
			lines.append(line.strip())

	w2v_embed_lst = []
	token2id, id2token = OrderedDict(), OrderedDict()
	for index, item in enumerate(lines):
		content = item.split()
		word = content[0]
		vector = [float(value) for value in content[1:]]
		w2v_embed_lst.append(vector)
		token2id[word] = index
		id2token[index] = word

	w2v_embed = np.asarray(w2v_embed_lst).astype(np.float32)

	return w2v_embed, token2id, id2token


