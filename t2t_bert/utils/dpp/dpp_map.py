import numpy as np
import math
import numpy as np
import numpy

def fast_map_dpp(kernel_matrix, max_length, epsilon=1E-10):
	"""
	Our proposed fast implementation of the greedy algorithm
	:param kernel_matrix: 2-d array
	:param max_length: positive int
	:param epsilon: small positive scalar
	:return: list
	reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
	paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
	"""
	item_size = kernel_matrix.shape[0]
	cis = np.zeros((max_length, item_size))
	di2s = np.copy(np.diag(kernel_matrix))
	selected_items = list()
	selected_item = np.argmax(di2s)
	selected_items.append(selected_item)
	while len(selected_items) < max_length:
		k = len(selected_items) - 1
		ci_optimal = cis[:k, selected_item]
		di_optimal = math.sqrt(di2s[selected_item])
		elements = kernel_matrix[selected_item, :]
		eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
		cis[k, :] = eis
		di2s -= np.square(eis)
		selected_item = np.argmax(di2s)
		if di2s[selected_item] < epsilon:
			break
		selected_items.append(selected_item)
	return selected_items

def greedy_map_dpp(L):
	"""
	greedy map
	reference: http://jgillenw.com/dpp-map.html
	paper: Near-Optimal MAP Inference for Determinantal Point Processes
	"""
	C = [];
	N = L.shape[0]
	U = list(range(0, N))
	num_left = N
	
	while len(U) > 0:
		scores = np.diag(L)
		# Select the max-scoring addition to the chosen set.
		max_loc = np.argmax(scores)
		max_score = scores[max_loc]
		
		if max_score < 1:
			break
		C.append(U[max_loc])
		del U[max_loc]

		# Compute the new kernel, conditioning on the current selection.
		inc_ids = list(range(0, max_loc))+list(range(max_loc+1, num_left))

		L = numpy.linalg.inv(L+np.diag([1]*(max_loc)+[0]+[1]*(num_left-max_loc-1)))
		num_left -= 1
		L = numpy.linalg.inv(L[np.ix_(inc_ids, inc_ids)]) - np.eye(num_left)
		
	return C
