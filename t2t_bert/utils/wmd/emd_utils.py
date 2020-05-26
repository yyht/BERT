import numpy as np
from pyemd import emd


def wmd_distance(w2v_model, document1, document2, distance_metric=None):
	len_pre_oov1 = len(document1)
	len_pre_oov2 = len(document2)
	document1 = [token for token in document1 if token in w2v_model]
	document2 = [token for token in document2 if token in w2v_model]
	diff1 = len_pre_oov1 - len(document1)
	diff2 = len_pre_oov2 - len(document2)

	dictionary = list(set(document1+document2))
	vocab_len = len(dictionary)

	if vocab_len == 1:
		# Both documents are composed by a single unique token
		return 0.0

	# Sets for faster look-up.
	docset1 = set(document1)
	docset2 = set(document2)

	# Compute distance matrix.
	distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
	for i, t1 in enumerate(dictionary):
		if t1 not in docset1:
			continue
		for j, t2 in enumerate(dictionary):
			if t2 not in docset2 or distance_matrix[i, j] != 0.0:
				continue
			if distance_metric == 'euclidean':
				# Compute Euclidean distance between word vectors.
				euclidean_distance = sqrt(np.sum((w2v_model[t1] - w2v_model[t2])**2))
				distance_matrix[i, j] = distance_matrix[j, i] = euclidean_distance
			elif distance_metric == 'cosine':
				t1_norm = np.sqrt(np.sum(np.power((w2v_model[t1]), 2)))
				t2_norm = np.sqrt(np.sum(np.power((w2v_model[t2]), 2)))
				cos_distance = np.sum(w2v_model[t1]*w2v_model[t2]) / (t1_norm*t2_norm+1e-10) 
				distance_matrix[i, j] = distance_matrix[j, i] = 1 - cos_distance
			else:
				euclidean_distance = np.sqrt(np.sum((w2v_model[t1] - w2v_model[t2])**2))
				distance_matrix[i, j] = distance_matrix[j, i] = euclidean_distance
	
	if np.sum(distance_matrix) == 0.0:
		# `emd` gets stuck if the distance matrix contains only zeros.
		return 1e-10

	keys = dict((e[1], e[0]) for e in enumerate(dictionary))
	def nbow(document):
		d = np.zeros(vocab_len, dtype=np.double)
		for word in document:
			d[keys[word]] += 1
		doc_len = len(document)
		for idx, freq in enumerate(d):
			d[idx] = freq / float(doc_len)  # Normalized word frequencies.
		return d

	# Compute nBOW representation of documents.
	d1 = nbow(document1)
	d2 = nbow(document2)

	return emd(d1, d2, distance_matrix)