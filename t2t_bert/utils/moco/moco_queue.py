import tensorflow as tf

def update_model_via_ema(
    model, ema_model, momentum, just_trainable_vars=False
):
    iterable = (
        zip(model.trainable_variables, ema_model.trainable_variables)
        if just_trainable_vars
        else zip(model.variables, ema_model.variables)
    )
    for p, p2 in iterable:
        p2.assign(momentum * p2 + (1.0 - momentum) * p)

class MoCoQueue:
	def __init__(self, embedding_dim, max_queue_length):
		self.embedding_dim = embedding_dim
		# Put a single zeros key in there to start with, it will be pushed out eventually
		with tf.device("CPU:0"):
			self.keys = tf.random.normal([2, self.embedding_dim])
		self.max_queue_length = max_queue_length

	def enqueue(self, new_keys):
		self.keys = tf.concat([new_keys, self.keys], 0)
		if self.keys.shape[0] > self.max_queue_length:
			self.keys = self.keys[:self.max_queue_length]