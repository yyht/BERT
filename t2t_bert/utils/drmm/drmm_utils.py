from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Permute, Dense, Dropout, Embedding, \
Flatten, Input, merge, Lambda, Reshape
from keras import backend
import tensorflow as tf



def pos_softmax(pos_neg_scores):
	exp_pos_neg_scores = [tf.exp(s) for s in pos_neg_scores]
	denominator = tf.add_n(exp_pos_neg_scores)
	return exp_pos_neg_scores[0] / denominator

def _kmax(x, top_k):
	return tf.nn.top_k(x, k=top_k, sorted=True, name=None)[0]

def _kmax_context(inputs, top_k):
	x, context_input = inputs
	vals, idxs = tf.nn.top_k(x, k=top_k, sorted=True)
	# hack that requires the context to have the same shape as similarity matrices
	# https://stackoverflow.com/questions/41897212/how-to-sort-a-multi-dimensional-tensor-using-the-returned-indices-of-tf-nn-top-k
	shape = tf.shape(x)
	mg = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape[:(x.get_shape().ndims - 1)]) + [top_k])], indexing='ij')
	val_contexts = tf.gather_nd(context_input, tf.stack(mg[:-1] + [idxs], axis=-1))

def _multi_kmax_concat(x, top_k, poses):
	slice_mats=list()
	for p in poses:
		slice_mats.append(tf.nn.top_k(tf.slice(x, [0,0,0], [-1,-1,p]), k=top_k, sorted=True, name=None)[0])
	concat_topk_max = tf.concat(slice_mats, -1, name='concat')
	return concat_topk_max

def _multi_kmax_context_concat(inputs, top_k, poses):
	x, context_input = inputs
	idxes, topk_vs = list(), list()
	for p in poses:
		val, idx = tf.nn.top_k(tf.slice(x, [0,0,0], [-1,-1, p]), k=top_k, sorted=True, name=None)
		topk_vs.append(val)
		idxes.append(idx)
	concat_topk_max = tf.concat(topk_vs, -1, name='concat_val')
	concat_topk_idx = tf.concat(idxes, -1, name='concat_idx')
	# hack that requires the context to have the same shape as similarity matrices
	# https://stackoverflow.com/questions/41897212/how-to-sort-a-multi-dimensional-tensor-using-the-returned-indices-of-tf-nn-top-k
	shape = tf.shape(x)
	mg = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape[:(x.get_shape().ndims - 1)]) + [top_k*len(poses)])], indexing='ij')
	val_contexts = tf.gather_nd(context_input, tf.stack(mg[:-1] + [concat_topk_idx], axis=-1))
	return tf.concat([concat_topk_max, val_contexts], axis=-1)
	# return backend.concatenate([concat_topk_max, val_contexts])

def _cov_dsim_layers(dim_sim, len_query, n_grams, n_filter, top_k, poses, selecter):
	re_input = Reshape((len_query, dim_sim, 1), name='ql_ds_doc')
	re_input = tf.reshape(len_query, )
	cov_sim_layers = dict()
	pool_sdim_layer=dict()
	pool_sdim_layer_context=dict()
	re_ql_ds=dict()
	pool_filter_layer=dict()
	for n_query, n_doc in n_grams:
		subsample_docdim = 1
		if selecter in ['strides']:
			subsample_docdim = n_doc
		dim_name = self._get_dim_name(n_query,n_doc)
		cov_sim_layers[dim_name] = \
		Conv2D(n_filter, kernel_size=(n_query, n_doc), strides=(1, subsample_docdim), padding="same", use_bias=True,\
				name='cov_doc_%s'%dim_name, kernel_initializer='glorot_uniform', activation='relu', \
				bias_constraint=None, kernel_constraint=None, data_format=None, bias_regularizer=None,
				activity_regularizer=None, weights=None, kernel_regularizer=None)

		pool_sdim_layer[dim_name] = Lambda(lambda x: self._multi_kmax_concat(x, top_k, poses), \
				name='ng_max_pool_%s_top%d_pos%d'%(dim_name, top_k, len(poses)))
		pool_sdim_layer_context[dim_name] = \
				Lambda(lambda x: self._multi_kmax_context_concat(x, top_k, poses), \
				name='ng_max_pool_%s_top%d__pos%d_context'%(dim_name, top_k, len(poses))) 
		re_ql_ds[dim_name] = Lambda(lambda t:backend.squeeze(t,axis=2), name='re_ql_ds_%s'%(dim_name))
		pool_filter_layer[dim_name] = \
				MaxPooling2D(pool_size=(1, n_filter), strides=None, padding='valid', data_format=None, \
				name='max_over_filter_doc_%s'%dim_name)

	ex_filter_layer = Permute((1, 3, 2), input_shape=(len_query, 1, n_filter), name='permute_filter_lenquery') 
	return re_input, cov_sim_layers, pool_sdim_layer, pool_sdim_layer_context, pool_filter_layer, ex_filter_layer, re_ql_ds
