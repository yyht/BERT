import tensorflow as tf

def sample_gumbel(shape, samples=1, eps=1e-20): 
	"""Sample from Gumbel(0, 1)"""
	if samples > 1:
		sample_shape = shape + [samples]
	else:
		sample_shape = shape
	U = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
	# return -tf.log(-tf.log(U + eps) + eps)
	return -tf.log(-tf.log(U))

def gumbel_softmax(logits, temperature=0.1, gumbel_samples=None,
				samples=1,
				exclude_mask=None):
	if gumbel_samples is None:
		gumbel_samples = sample_gumbel(tf.shape(logits), samples=samples)

	gumbel_softmax_sample = logits + 
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)




