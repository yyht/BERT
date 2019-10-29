def prelln_transformer_model(input_tensor,
						attention_mask=None,
						hidden_size=768,
						num_hidden_layers=12,
						num_attention_heads=12,
						intermediate_size=3072,
						intermediate_act_fn=gelu,
						hidden_dropout_prob=0.1,
						attention_probs_dropout_prob=0.1,
						initializer_range=0.02,
						do_return_all_layers=False,
						shared_type=None,
						adapter_fn=None):
	"""Multi-headed, multi-layer Transformer from "Attention is All You Need".

	This is almost an exact implementation of the original Transformer encoder.

	See the original paper:
	https://arxiv.org/abs/1706.03762

	Also see:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
		attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
			seq_length], with 1 for positions that can be attended to and 0 in
			positions that should not be.
		hidden_size: int. Hidden size of the Transformer.
		num_hidden_layers: int. Number of layers (blocks) in the Transformer.
		num_attention_heads: int. Number of attention heads in the Transformer.
		intermediate_size: int. The size of the "intermediate" (a.k.a., feed
			forward) layer.
		intermediate_act_fn: function. The non-linear activation function to apply
			to the output of the intermediate/feed-forward layer.
		hidden_dropout_prob: float. Dropout probability for the hidden layers.
		attention_probs_dropout_prob: float. Dropout probability of the attention
			probabilities.
		initializer_range: float. Range of the initializer (stddev of truncated
			normal).
		do_return_all_layers: Whether to also return all layers or just the final
			layer.

	Returns:
		float Tensor of shape [batch_size, seq_length, hidden_size], the final
		hidden layer of the Transformer.

	Raises:
		ValueError: A Tensor shape or parameter is invalid.
	"""
	if hidden_size % num_attention_heads != 0:
		raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))

	attention_head_size = int(hidden_size / num_attention_heads)

	input_shape = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	input_width = input_shape[2]

	# The Transformer performs sum residuals on all layers so the input needs
	# to be the same as the hidden size.
	if input_width != hidden_size:
		raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
										 (input_width, hidden_size))

	# We keep the representation as a 2D tensor to avoid re-shaping it back and
	# forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
	# the GPU/CPU but may not be free on the TPU, so we want to minimize them to
	# help the optimizer.
	prev_output = bert_utils.reshape_to_matrix(input_tensor)

	all_layer_outputs = []

	def layer_scope(idx, shared_type):
		if shared_type == 'all':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			} 
		elif shared_type == 'attention':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate_{}'.format(idx),
				'output':'output_{}'.format(idx)
			}
		elif shared_type == 'ffn':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention_{}'.format(idx),
				'intermediate':'intermediate',
				'output':'output'
			}
		else:
			tmp = {
				"layer":"layer_{}".format(idx),
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			}

		return tmp

	all_layer_outputs = []

	for layer_idx in range(num_hidden_layers):

		idx_scope = layer_scope(layer_idx, shared_type)

		with tf.variable_scope(idx_scope['layer'], reuse=tf.AUTO_REUSE):
			layer_input = prev_output

			with tf.variable_scope(idx_scope['attention'], reuse=tf.AUTO_REUSE):
				attention_heads = []

				with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
					layer_input_pre = layer_norm(layer_input)

				with tf.variable_scope("self"):
					[attention_head, 
					attention_scores] = attention_layer(
							from_tensor=layer_input_pre,
							to_tensor=layer_input_pre,
							attention_mask=attention_mask,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length)
					attention_heads.append(attention_head)

				attention_output = None
				if len(attention_heads) == 1:
					attention_output = attention_heads[0]
				else:
					# In the case where we have other sequences, we just concatenate
					# them to the self-attention head before the projection.
					attention_output = tf.concat(attention_heads, axis=-1)

				# Run a linear projection of `hidden_size` then add a residual
				# with `layer_input`.
				with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
					attention_output = tf.layers.dense(
							attention_output,
							hidden_size,
							kernel_initializer=create_initializer(initializer_range))
					attention_output = dropout(attention_output, hidden_dropout_prob)

					# attention_output = layer_norm(attention_output + layer_input)
					attention_output = attention_output + layer_input

			with tf.variable_scope(idx_scope['output'], reuse=tf.AUTO_REUSE):
				attention_output_pre = layer_norm(attention_output)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope(idx_scope['intermediate'], reuse=tf.AUTO_REUSE):
				intermediate_output = tf.layers.dense(
						attention_output_pre,
						intermediate_size,
						activation=intermediate_act_fn,
						kernel_initializer=create_initializer(initializer_range))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope(idx_scope['output'], reuse=tf.AUTO_REUSE):
				layer_output = tf.layers.dense(
						intermediate_output,
						hidden_size,
						kernel_initializer=create_initializer(initializer_range))
				layer_output = dropout(layer_output, hidden_dropout_prob)

				# layer_output = layer_norm(layer_output + attention_output)
				layer_output = layer_output + attention_output
				prev_output = layer_output
				all_layer_outputs.append(layer_output)

	if do_return_all_layers:
		final_outputs = []
		for layer_output in all_layer_outputs:
			final_output = bert_utils.reshape_from_matrix(layer_output, input_shape)
			final_outputs.append(final_output)
		return final_outputs
	else:
		final_output = bert_utils.reshape_from_matrix(prev_output, input_shape)
		return final_outputs
