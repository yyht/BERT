import tensorflow as tf
import numpy as np
from collections import OrderedDict

def load_pretrained_w2v(vocab_path, w2v_path):
	with tf.gfile.Open(w2v_path) as frobj:
		vector = []
		for index, line in enumerate(frobj):
			if index == 0:
				vocab_size, vector_size = map(int, line.strip().split())
				continue
			vector.append(line.strip())

	with tf.gfile.Open(vocab_path) as frobj:
		vocab = []
		for line in frobj:
			vocab.append(line.strip())

	w2v = {}
	for item in vector:
		content = item.split()
		w2v[content[0]] = map(float, content[1:])

	w2v_embed_lst = []
	token2id, id2token = OrderedDict(), OrderedDict()
	for index, word in enumerate(vocab):
		w2v_embed_lst.append(w2v[word])
		token2id[word] = index
		id2token[index] = word

	w2v_embed = np.asarray(w2v_embed_lst).astype(np.float32)

	return w2v_embed, token2id, id2token


