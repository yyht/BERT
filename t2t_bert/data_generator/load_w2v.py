import tensorflow as tf
import numpy as np
from collections import OrderedDict

def load_pretrained_w2v(vocab_path, w2v_path, vector_size=None):
	try:
		with tf.gfile.Open(w2v_path, "r") as frobj:
			header = frobj.readline()
			vocab_size, vector_size = map(int, header.split())

			vector = []
			for index in range(vocab_size):
				vector.append(frobj.readline().strip())

		print("==size of vocab of w2v==", vocab_size, vector_size)
	except:
		vector = []
		print("==random initialiaztion==")

	with tf.gfile.Open(vocab_path, "r") as frobj:
		vocab = []
		for line in frobj:
			vocab.append(line.strip())

	vocab_size = len(vocab)
	print("==actual corpus based vocab size==", len(vocab))
	tf.logging.info("***** vocab size *****", str(vocab_size))

	if vector:
		w2v = {}
		for item in vector:
			content = item.split()
			w2v[content[0]] = [float(vec) for vec in content[1:]]
	else:
		w2v = {}

	if not vector_size:
		vector_size = 96

	w2v_embed_lst = []
	token2id, id2token = OrderedDict(), OrderedDict()
	for index, word in enumerate(vocab):
		if word in w2v:
			vector_size = len(w2v[word])
			w2v_embed_lst.append(w2v[word])
		else:
			w2v_embed_lst.append(np.random.uniform(low=-0.1, high=0.1, 
								size=(vector_size,)).astype(np.float32).tolist())

		token2id[word] = index
		id2token[index] = word
	w2v_embed = np.asarray(w2v_embed_lst).astype(np.float32)
	if vocab_size == len(vocab):
		is_extral_symbol = 0
	else:
		is_extral_symbol = 1

	if w2v:
		use_pretrained = True
	else:
		use_pretrained = False

	return w2v_embed, token2id, id2token, is_extral_symbol, use_pretrained


