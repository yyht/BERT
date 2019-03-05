import tensorflow as tf
import numpy as np
from collections import OrderedDict

def load_pretrained_w2v(vocab_path, w2v_path):
	with tf.gfile.Open(w2v_path, "r") as frobj:
		header = frobj.readline()
		vocab_size, vector_size = map(int, header.split())

		vector = []
		for index in range(vocab_size):
			vector.append(frobj.readline().strip())

	with tf.gfile.Open(vocab_path, "r") as frobj:
		vocab = []
		for index in range(vocab_size):
			vocab.append(frobj.readline().strip())

	w2v = {}
	for item in vector:
		content = item.split()
		w2v[content[0]] = [float(vec) for vec in content[1:]]

	w2v_embed_lst = []
	token2id, id2token = OrderedDict(), OrderedDict()
	for index, word in enumerate(vocab):
		w2v_embed_lst.append(w2v[word])
		token2id[word] = index
		id2token[index] = word

	w2v_embed = np.asarray(w2v_embed_lst).astype(np.float32)

	return w2v_embed, token2id, id2token


