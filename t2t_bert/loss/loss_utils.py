import tensorflow as tf
import numpy as np

EPSILON = 1e-30

def focal_loss_binary_v2(config, logits, labels):
	"""
	alpha = 0.5
	gamma = 2.0
	"""
	alpha = config.get("alpha", 0.5)
	gamma = config.get("gamma", 2.0)

	labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

	predictions = tf.nn.softmax(logits)
	batch_idxs = tf.range(0, tf.shape(labels)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	print(batch_idxs.get_shape(), labels.get_shape(), '=======focal loss shape=====')

	idxs = tf.concat([batch_idxs, labels], 1)
	y_true_pred = tf.gather_nd(predictions, idxs)

	labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

	postive_loss = labels * tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)* alpha
	negative_loss = (1-labels)*tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma) * (1 - alpha)

	losses = -postive_loss - negative_loss
	# return tf.reduce_mean(losses), predictions
	return losses, predictions

def label_smoothing(inputs, epsilon=0.1):
	'''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
	inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
	epsilon: Smoothing rate.
	
	For example,
	
	```
	import tensorflow as tf
	inputs = tf.convert_to_tensor([[[0, 0, 1], 
	   [0, 1, 0],
	   [1, 0, 0]],
	  [[1, 0, 0],
	   [1, 0, 0],
	   [0, 1, 0]]], tf.float32)
	   
	outputs = label_smoothing(inputs)
	
	with tf.Session() as sess:
		print(sess.run([outputs]))
	
	>>
	[array([[[ 0.03333334,  0.03333334,  0.93333334],
		[ 0.03333334,  0.93333334,  0.03333334],
		[ 0.93333334,  0.03333334,  0.03333334]],
	   [[ 0.93333334,  0.03333334,  0.03333334],
		[ 0.93333334,  0.03333334,  0.03333334],
		[ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
	```    
	'''
	V = inputs.get_shape().as_list()[-1] # number of channels
	return ((1-epsilon) * inputs) + (epsilon / V)

def ce_label_smoothing(config, logits, labels):
	gamma = config.get("gamma", 2.0)

	log_probs = tf.nn.log_softmax(logits, axis=-1)

	labels = tf.reshape(labels, [-1])
	one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)

	smoothed_label = label_smoothing(one_hot_labels)
	per_example_loss = -tf.reduce_sum(smoothed_label * log_probs, axis=-1)
	
	return per_example_loss

def focal_loss_multi_v1(config, logits, labels):
	gamma = config.get("gamma", 2.0)

	labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

	predictions = tf.exp(tf.nn.log_softmax(logits, axis=-1))

	batch_idxs = tf.range(0, tf.shape(labels)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	idxs = tf.concat([batch_idxs, labels], 1)
	y_true_pred = tf.gather_nd(predictions, idxs)

	labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

	losses =  tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)

	return -losses, y_true_pred

def class_balanced_focal_loss_multi_v1(config, logits, labels, label_weights):
	gamma = config.get("gamma", 2.0)

	class_balanced_weights = tf.gather(label_weights, labels)

	labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

	predictions = tf.exp(tf.nn.log_softmax(logits, axis=-1))

	batch_idxs = tf.range(0, tf.shape(labels)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	idxs = tf.concat([batch_idxs, labels], 1)
	y_true_pred = tf.gather_nd(predictions, idxs)

	losses =  tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma) * class_balanced_weights

	return -losses, predictions

def weighted_loss_ratio(config, losses, labels, ratio_weight):
	unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
	appear_times = tf.gather(unique_count, unique_idx)
	# appear_times = tf.reshape(appear_times, [-1, 1])

	weighted_loss = losses * ratio_weight
	weighted_loss = weighted_loss / tf.cast((EPSILON+appear_times), tf.float32)

	return weighted_loss, None

def sparse_amsoftmax_loss(config, logits, labels, **kargs):
	"""
	scale = 30,
	margin = 0.35
	"""
	config = args[0]
	scale = config.scale
	margin =config.margin

	labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

	y_pred = tf.nn.l2_normalize(logits, axis=-1)
	batch_idxs = tf.range(0, tf.shape(labels)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	idxs = tf.concat([batch_idxs, labels], 1)
	y_true_pred = tf.gather_nd(y_pred, idxs)

	y_true_pred = tf.expand_dims(y_true_pred, 1)
	y_true_pred_margin = y_true_pred - margin
	_Z = tf.concat([y_pred, y_true_pred_margin], 1) 
	_Z = _Z * scale 
	logZ = tf.reduce_logsumexp(_Z, 1, keepdims=True)
	logZ = logZ + tf.log(1 - tf.exp(scale * y_true_pred - logZ) + EPSILON)
	losses = y_true_pred_margin * scale - logZ
	# return -tf.reduce_mean(losses), y_pred
	return -losses, y_pred

def center_loss_v1(config, embedding, labels, **kargs):
	'''
	embedding dim : (batch_size, num_features)
	'''
	num_features = embedding.get_shape()[-1]
	with tf.variable_scope(config.scope+"_center_loss"):
		centroids = tf.get_variable('center',
						shape=[config.num_classes, num_features],
						dtype=tf.float32,
						initializer=tf.contrib.layers.xavier_initializer(),
						trainable=False)

		centroids_delta = tf.get_variable('centroidsUpdateTempVariable',
						shape=[config.num_classes, num_features],
						dtype=tf.float32,
						initializer=tf.zeros_initializer(),
						trainable=False)

		centroids_batch = tf.gather(centroids, labels)
		# cLoss = tf.nn.l2_loss(embedding - centroids_batch) / (batch_size) # Eq. 2
		
		# cLoss = tf.reduce_mean(tf.reduce_sum((embedding - centroids_batch)**2, axis=-1))
		cLoss = tf.reduce_sum((embedding - centroids_batch)**2, axis=-1)

		diff = centroids_batch - embedding

		delta_c_nominator = tf.scatter_add(centroids_delta, labels, diff)
		indices = tf.expand_dims(labels, -1)
		updates = tf.cast(tf.ones_like(labels), tf.float32)
		shape = tf.constant([num_features])

		labels_sum = tf.expand_dims(tf.scatter_nd(indices, updates, shape),-1)
		centroids = centroids.assign_sub(config.alpha * delta_c_nominator / (1.0 + labels_sum))

		centroids_delta = centroids_delta.assign(tf.zeros([config.num_classes, num_features]))

		return cLoss, centroids

def center_loss_v2(config, features, labels, centers=None, **kargs):
	alpha = config.alpha
	num_classes = config.num_classes
	with tf.variable_scope(config.scope+"_center_loss"):
		print("==center loss==")
		len_features = features.get_shape()[1]
		if not centers:
			centers = tf.get_variable('centers', 
							[num_classes, len_features], 
							dtype=tf.float32,
							initializer=tf.contrib.layers.xavier_initializer(),
							trainable=False)
			print("==add center parameters==")
	 
		centers_batch = tf.gather(centers, labels)

		loss = tf.nn.l2_loss(features - centers_batch)
	 
		diff = centers_batch - features
	 
		unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
		appear_times = tf.gather(unique_count, unique_idx)
		appear_times = tf.reshape(appear_times, [-1, 1])
	 
		diff = diff / tf.cast((1 + appear_times), tf.float32)
		diff = alpha * diff

		centers_update_op = tf.scatter_sub(centers, labels, diff)

		tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)
		
		return loss, centers

def spread_loss(config, labels, activations, margin, **kargs):
	activations_shape = activations.get_shape().as_list()
	mask_t = tf.equal(labels, 1)
	mask_i = tf.equal(labels, 0)    
	activations_t = tf.reshape(
	  tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
	)    
	activations_i = tf.reshape(
	  tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
	)    
	gap_mit = tf.reduce_sum(tf.square(tf.nn.relu(margin - (activations_t - activations_i))))
	return gap_mit   

def margin_loss(config, y, preds, **kargs):    
	y = tf.cast(y,tf.float32)
	loss = y * tf.square(tf.maximum(0., 0.9 - preds)) + \
		0.25 * (1.0 - y) * tf.square(tf.maximum(0., preds - 0.1))
	# loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))  
	loss = tf.reduce_sum(loss, axis=1)
	return loss


def multi_label_loss(config, logits, labels, *args, **kargs):
	loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=logits, 
					labels=labels)
	# loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
	loss = tf.reduce_sum(loss, axis=1)
	return loss

def multi_label_hot(config, prediction, **kargs):
	threshold = kargs["threshold"]
	prediction = tf.cast(prediction, tf.float32)
	threshold = float(threshold)
	pred_label = tf.cast(tf.geater(prediction, threshold), tf.int32)

	return pred_label




