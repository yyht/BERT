import tensorflow as tf
from utils.bert import bert_utils

class FlipGradientBuilder(object):
	def __init__(self):
		self.num_calls = 0

	def __call__(self, x, l=1.0):
		grad_name = "FlipGradient%d" % self.num_calls
		@ops.RegisterGradient(grad_name)
		def _flip_gradients(op, grad):
			return [tf.negative(grad) * l]
		
		g = tf.get_default_graph()
		with g.gradient_override_map({"Identity": grad_name}):
			y = tf.identity(x)
			
		self.num_calls += 1
		return y
	
flip_gradient = FlipGradientBuilder()

def check_tf_version():
	version = tf.__version__
	print("==tf version==", version)
	if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
		return True
	else:
		return False

def sample_gumbel(shape, samples=1, eps=1e-20): 
	"""Sample from Gumbel(0, 1)"""
	if samples > 1:
		sample_shape = shape + [samples]
	else:
		sample_shape = shape
	U = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
	# return -tf.log(-tf.log(U + eps) + eps)
	return -tf.log(-tf.log(U))

def gumbel_softmax(logits, origin_input, 
				output_target_mapping, 
				output_is_target,
				vocab_size,
				temperature=0.1, 
				gumbel_samples=None,
				samples=1,
				sample_type='argmax',
				if_flip_grad=True):

	# logits: [batch_size, num_predict, vocab_size]
	# output_target_mapping: [batch_size, num_predict, seq_length]

	if gumbel_samples is None:
		gumbel_samples = sample_gumbel(tf.shape(logits), samples=samples)

	# gumbel_softmax_sample: [batch_size, num_predict, vocab_size]
	gumbel_softmax_sample = logits + gumbel_samples
	if sample_type == 'argmax':
		# samples: [batch_size, num_predict]
		samples = tf.argmax(gumbel_softmax_sample, -1)
		# output_target_mapping: [batch_size, num_predict, seq_length]
		if check_tf_version():
			samples = tf.einsum("...m,...ml->...l",
						  tf.cast(samples, tf.float32),
						  tf.cast(output_target_mapping, tf.float32))
		else:
			samples = tf.einsum("am,aml->al",
						  tf.cast(samples, tf.float32),
						  tf.cast(output_target_mapping, tf.float32))
		# samples: [batch_size, seq_length]
		sample_input = tf.where(output_is_target, samples, origin_input)
		sample_input = tf.cast(sample_input, tf.int64)
	elif sample_type == 'straight_though':
		input_shape_list = bert_utils.get_shape_list(gumbel_softmax_sample, expected_rank=3)
		width = input_shape_list[2]
		# gumbel_softmax_sample: [batch_size x num_predict, vocab_size]
		gumbel_softmax_sample = tf.reshape(gumbel_softmax_sample, [-1, vocab_size])
		sampled_logprob_temp = tf.exp(tf.nn.log_softmax(gumbel_softmax_sample/temperature, axis=1))
		# sampled_hard_id: [batch_size x num_predict, vocab_size]
		sampled_hard_id = tf.one_hot(tf.argmax(gumbel_softmax_sample, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id
		if if_flip_grad:
			tf.logging.info("****** apply gradient flipping *******")
			sampled_logprob_temp = flip_gradient(sampled_logprob_temp)
		else:
			tf.logging.info("****** not apply gradient flipping *******")

		# [batch_size x num_predict, vocab_size]
		samples = tf.stop_gradient(sampled_hard_id-sampled_logprob_temp) + (sampled_logprob_temp)
		# [batch_size, num_predict, vocab_size]
		samples = tf.reshape(input_shape_list, samples)
		samples = tf.einsum("amb,aml->alb",
						  tf.cast(samples, tf.float32),
						  tf.cast(output_target_mapping, tf.float32))
		origin_input_shape_list = bert_utils.get_shape_list(origin_input, expected_rank=2)
		origin_input = tf.reshape([-1], origin_input)
		# [batch_size x seq_length, vocab_size]
		origin_input_onehot = tf.one_hot(origin_input_onehot, axis=1, dtype=tf.float32)
		# [batch_size , seq_length, vocab_size]
		origin_input = tf.reshape(origin_input_shape_list+[-1], origin_input_onehot)
		# [batch_size , seq_length, 1]
		output_is_target = tf.expand_dims(output_is_target, axis=-1)
		# [batch_size , seq_length, vocab_size]
		sample_input = (1 - output_is_target) * origin_input + output_is_target * samples

	return sample_input

  