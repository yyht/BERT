import tensorflow as tf
import numpy as np
from model.gpt import gpt_utils
from utils.bert.bert_utils import get_shape_list

def top_k_logits(logits, k):
	if k == 0:
		# no truncation
		return logits

	def _top_k():
		values, _ = tf.nn.top_k(logits, k=k)
		min_values = values[:, -1, tf.newaxis]
		return tf.where(
			logits < min_values,
			tf.ones_like(logits, dtype=logits.dtype) * -1e10,
			logits,
		)
	return tf.cond(
	   tf.equal(k, 0),
	   lambda: logits,
	   lambda: _top_k(),
	)

def sample_sequence(encoder_decoder, hparams, 
				length, start_token=None, 
				batch_size=None, context=None, 
				temperature=1, top_k=0):
	if start_token is None:
		assert context is not None, 'Specify exactly one of start_token and context!'
	else:
		assert context is None, 'Specify exactly one of start_token and context!'
		context = tf.fill([batch_size, 1], start_token)

	# batch_size, seq_len = get_shape_list(context, 2)
	batch_size = context.get_shape()[0]

	def step(hparams, tokens, past=None):
		features = {}
		features['input_ids'] = tokens
		features['past'] = past
		
		inference_model = encoder_decoder(hparams, features, 
										[], tf.estimator.ModeKeys.PREDICT, 
										"", reuse=tf.AUTO_REUSE)

		logits = inference_model.get_sequence_output_logits()[:, :, :hparams.n_vocab]
		presents = inference_model.get_present()
		# presents = tf.reshape(presents, get_shape_list(presents, 6))
		presents.set_shape(gpt_utils.past_shape(hparams=hparams, batch_size=batch_size))
		return {
			'logits': logits,
			'presents': presents,
		}

	with tf.name_scope('sample_sequence'):
		# Don't feed the last context token -- leave that to the loop below
		# TODO: Would be slightly faster if we called step on the entire context,
		# rather than leaving the last token transformer calculation to the while loop.
		context_output = step(hparams, context[:, :-1])

		def get_samples_logits(samples, logits):
			batch_idxs = tf.range(0, tf.shape(samples)[0])
			batch_idxs = tf.expand_dims(batch_idxs, 1)

			idxs = tf.concat([batch_idxs, samples], 1)
			samples_logits = tf.expand_dims(tf.gather_nd(logits, idxs), -1)
			return samples_logits

		def body(past, prev, output, output_logits):
			next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
			logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
			my_logits = tf.nn.log_softmax(logits, axis=-1)
			logits = top_k_logits(logits, k=top_k)
			samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
			samples_logits = get_samples_logits(samples, my_logits)
			
			return [
				tf.concat([past, next_outputs['presents']], axis=-2),
				tf.squeeze(samples, axis=[1]),
				tf.concat([output, samples], axis=1),
				tf.concat([output_logits, samples_logits], axis=1),
			]

		def cond(*args):
			return True

		_, _, tokens, logits = tf.while_loop(
			cond=cond, body=body,
			maximum_iterations=length,
			loop_vars=[
				context_output['presents'],
				context[:, -1],
				context,
				tf.cast(tf.zeros_like(context), tf.float32)
			],
			shape_invariants=[
				tf.TensorShape(gpt_utils.past_shape(hparams=hparams, batch_size=batch_size)),
				# tf.TensorShape([batch_size,  hparams.n_layer, 2, hparams.n_head, None, hparams.n_embd // hparams.n_head]),
				tf.TensorShape([batch_size]),
				tf.TensorShape([batch_size, None]),
				tf.TensorShape([batch_size, None]),
			],
			back_prop=False,
		)

		return {
			"logits":logits,
			"tokens":tokens
		}
