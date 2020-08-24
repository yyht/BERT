import tensorflow as tf
import numpy as np

from utils.funnel_transformer import funnel_transformer_ops_v1 as funnel_transformer_ops
from utils.funnel_transformer import tf_utils
from utils.textcnn import conv1d_transpose

from utils.bert import bert_utils

def check_tf_version():
	version = tf.__version__
	print("==tf version==", version)
	if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
		return True
	else:
		return False

def input_projection(net_config, input_embed, initializer):
	"""Project input embedding to a proper dimension if needed."""
	net_config = net_config
	ret_dict = {}

	output = input_embed
	if net_config.d_embed != net_config.d_model:
		tf.logging.info("Project input embedding: %s -> %s",
										net_config.d_embed, net_config.d_model)
		output = funnel_transformer_ops.dense(
				output,
				net_config.d_model,
				inp_shape=net_config.d_embed,
				initializer=initializer,
				scope="input_projection")

	return output, ret_dict

def get_embedding_table(net_config, scope="input", dtype=tf.float32):
	"""Get the corresponding embeeding table."""
	net_config = net_config
	with tf.variable_scope(scope, reuse=True):
		with tf.variable_scope("word_embedding", reuse=True):
			lookup_table = tf.get_variable(
					"lookup_table", [net_config.vocab_size, net_config.d_model],
					dtype=dtype)
	return lookup_table

##############################################
##### Pooling related section            #####
##############################################
def stride_pool(net_config, tensor, axis):
	"""Perform pooling by stride slicing the tensor along the axis."""
	if tensor is None:
		return None

	net_config = net_config
	pool_size = net_config.pooling_size
	if isinstance(tensor, (tuple, list)):
		ndims = tensor[0].shape.ndims
	else:
		ndims = tensor.shape.ndims
	axis = axis % ndims

	slice_list = []
	for i in range(ndims):
		if i == axis:
			if net_config.separate_cls:
				if net_config.truncate_seq:
					slice_list.append(slice(1, -1, pool_size))
				else:
					slice_list.append(slice(1, None, pool_size))
			else:
				slice_list.append(slice(None, None, pool_size))
			break
		else:
			slice_list.append(slice(None))

	if net_config.separate_cls:
		cls_slice_list = []
		for i in range(ndims):
			if i == axis:
				cls_slice_list.append(slice(None, 1))
				break
			else:
				cls_slice_list.append(slice(None))

	def _pool_func(origin):
		pooled = origin[slice_list]
		if net_config.separate_cls:
			pooled = tf.concat([origin[cls_slice_list], pooled], axis=axis)
		return pooled

	if isinstance(tensor, (tuple, list)):
		return list(map(_pool_func, tensor))
	else:
		return _pool_func(tensor)

def pool_tensor(net_config, tensor, mode="mean"):
	"""Apply 1D pooling to a tensor of size [B x T (x H)]."""
	if tensor is None:
		return None

	net_config = net_config
	ndims = tensor.shape.ndims
	pool_size = net_config.pooling_size

	if net_config.separate_cls:
		cls_tensor = tensor[:, :1]
		if net_config.truncate_seq:
			pooled = tensor[:, 1:-1]
		else:
			pooled = tensor[:, 1:]
	else:
		pooled = tensor

	if ndims == 2: pooled = pooled[:, :, None]
	if mode == "mean":
		if check_tf_version():
			pooled = tf.nn.avg_pool1d(
					pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using tf avg_pool1d")
		else:
			pooled = tf_utils.avg_pool1d(
					pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using my tf avg_pool1d")
	elif mode == "max":
		if check_tf_version():
			pooled = tf.nn.max_pool1d(
					pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using tf max_pool1d")
		else:
			pooled = tf_utils.max_pool1d(
					pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using my tf avg_pool1d")
	elif mode == "min":
		if check_tf_version():
			pooled = -tf.nn.max_pool1d(
					-pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using tf min_pool1d")
		else:
			pooled = -tf_utils.max_pool1d(
					-pooled,
					ksize=pool_size,
					strides=pool_size,
					data_format="NWC",
					padding="SAME")
			tf.logging.info(" using my tf min_pool1d")
	else:
		raise NotImplementedError
	if ndims == 2: pooled = tf.squeeze(pooled, 2)

	if net_config.separate_cls:
		pooled = tf.concat([cls_tensor, pooled], axis=1)

	return pooled

def rel_shift_pos_enc(net_config, q_len, q_pow, k_len, k_pow, is_training,
											dtype, name):
	"""Get positional encoding under the relative shift implementation."""
	net_config = net_config
	pool_size = net_config.pooling_size

	q_stride = pool_size ** q_pow
	k_stride = pool_size ** k_pow
	shift = q_stride // k_stride

	min_pos_k = 1 - k_stride
	max_pos_k = min_pos_k + (k_len - 1) * k_stride
	min_pos_q = 1 - q_stride

	ref_point = min_pos_q - min_pos_k
	num_to_remove = shift * q_len
	max_dist = ref_point + num_to_remove * k_stride
	min_dist = min_pos_q - max_pos_k
	rel_pos_id = tf.range(max_dist, min_dist - 1, -k_stride)

	enc = funnel_transformer_ops.get_pos_enc_gpu(
			rel_pos_id,
			net_config.d_model,
			net_config.dropout,
			is_training=is_training,
			dtype=dtype,
			name=name)

	pos_enc = (enc, shift)

	return pos_enc

def init_attn_structures(net_config, attn_structures, 
					hidden, seg_id, pos_id, is_training, name):
	"""Initialize extra structures needed for attention."""
	net_config = net_config
	if net_config.rel_attn_type == "null":
		attn_structures = (None, None, None)
	else:
		if attn_structures is None:
			print("==use new attention structures==")
			print(hidden, "===hiddens===")
			seq_len = tf.shape(hidden)[1]

			if net_config.rel_attn_type == "factorized":
				if pos_id is None:
					half_len = tf.cast(seq_len // 2, tf.float32)
					pos_id = tf.range(-half_len, half_len, 1.0)
				pos_enc = funnel_transformer_ops.get_pos_enc(
						pos_id,
						pos_id,
						net_config.d_model,
						net_config.dropout,
						is_training=is_training,
						dtype=hidden.dtype,
						name=name)
			elif net_config.rel_attn_type == "rel_shift":
				assert pos_id is None
				seq_len_fp = tf.cast(seq_len, tf.float32)
				rel_pos_id = tf.range(seq_len_fp, -seq_len_fp, -1.0)
				enc = funnel_transformer_ops.get_pos_enc_gpu(
						rel_pos_id,
						net_config.d_model,
						net_config.dropout,
						is_training=is_training,
						dtype=hidden.dtype,
						name=name)
				shift = 1
				pos_enc = (enc, shift)
			else:
				raise NotImplementedError
			seg_mat = funnel_transformer_ops.seg_id_to_mat(net_config, seg_id, seg_id)
			num_real_token = seq_len - 1
			func_mask = tf.pad(
					tf.ones([num_real_token, num_real_token], dtype=hidden.dtype),
					[[1, 0], [1, 0]])

			attn_structures = (pos_enc, seg_mat, func_mask)
			
	return attn_structures

def pre_attn_pooling(net_config, output, pos_enc, seg_mat, input_mask,
										 func_mask, block_idx, is_training, name):
	"""Perform pooling before the attention layer."""
	net_config = net_config
	if net_config.pool_q_only:
		seg_mat = stride_pool(net_config, seg_mat, 1)
		output = pool_tensor(net_config, output, mode=net_config.pooling_type)
		func_mask = stride_pool(net_config, func_mask, 0)
		if pos_enc is not None:
			if net_config.rel_attn_type == "factorized":
				pos_enc = stride_pool(net_config, pos_enc[:2], 0) + pos_enc[2:]
			elif net_config.rel_attn_type == "rel_shift":
				pos_enc = rel_shift_pos_enc(net_config,
						q_len=tf.shape(func_mask)[0], q_pow=block_idx,
						k_len=tf.shape(func_mask)[1], k_pow=block_idx-1,
						is_training=is_training, dtype=func_mask.dtype,
						name=name)
			else:
				raise NotImplementedError
	else:
		seg_mat = stride_pool(net_config, seg_mat, 1)
		seg_mat = stride_pool(net_config, seg_mat, 2)
		output = pool_tensor(net_config, output, mode=net_config.pooling_type)
		func_mask = stride_pool(net_config, func_mask, 0)
		func_mask = stride_pool(net_config, func_mask, 1)
		input_mask = pool_tensor(net_config, input_mask, mode="min")
		if pos_enc is not None:
			if net_config.rel_attn_type == "factorized":
				pos_enc = stride_pool(net_config, pos_enc, 0)
			elif net_config.rel_attn_type == "rel_shift":
				pos_enc = rel_shift_pos_enc(net_config,
						q_len=tf.shape(func_mask)[0], q_pow=block_idx,
						k_len=tf.shape(func_mask)[1], k_pow=block_idx,
						is_training=is_training, dtype=func_mask.dtype,
						name=name)
			else:
				raise NotImplementedError

	return output, pos_enc, seg_mat, input_mask, func_mask

def post_attn_pooling(net_config, pos_enc, seg_mat, input_mask, func_mask,
											block_idx, is_training, name):
	"""Perform pooling after the attention layer."""
	net_config = net_config
	if net_config.pool_q_only:
		seg_mat = stride_pool(net_config, seg_mat, 2)
		func_mask = stride_pool(net_config, func_mask, 1)
		input_mask = pool_tensor(net_config, input_mask, mode="min")
		if pos_enc is not None:
			if net_config.rel_attn_type == "factorized":
				pos_enc = pos_enc[:2] + stride_pool(net_config, pos_enc[2:], 0)
			elif net_config.rel_attn_type == "rel_shift":
				pos_enc = rel_shift_pos_enc(net_config,
						q_len=tf.shape(func_mask)[1], q_pow=block_idx,
						k_len=tf.shape(func_mask)[1], k_pow=block_idx,
						is_training=is_training, dtype=func_mask.dtype,
						name=name)
			else:
				raise NotImplementedError

	return pos_enc, seg_mat, input_mask, func_mask

##############################################
##### Upsampling related section         #####
##############################################
def upsample(net_config, output, stride, tgt_len):
	"""Upsample a hidden state by stride."""
	if stride == 1:
		return output

	net_config = net_config

	if net_config.separate_cls:
		cls_output = output[:, :1]
		output = output[:, 1:]

	if check_tf_version():
		output = tf.repeat(output, repeats=stride, axis=1)
	else:
		output = tf_utils.repeat(output, repeats=stride, axis=1)

	if net_config.separate_cls:
		if net_config.truncate_seq:
			pad_len = stride - 1
			output = tf.pad(output, [[0, 0], [0, pad_len], [0, 0]])
		else:
			output = output[:, :tgt_len - 1]
		output = tf.concat([cls_output, output], axis=1)

	return output

def bridge_layer(net_config, hiddens, input_mask,
								reuse=tf.AUTO_REUSE):
	"""A bridge layer between encoder and decoder."""
	net_config = net_config
	ret_dict = {}
	if net_config.get("tgt_len", None):
		tgt_len = net_config['tgt_len']
	else:
		tgt_len = tf.shape(input_mask)[1]
	with tf.variable_scope("upsampling_layer", reuse=reuse):
		# upsample hiddens based on the provided block indices
		upsampled_hids = []
		cum_num_layer = 0
		for block_idx in range(net_config.n_block):
			stride = 2 ** block_idx
			cum_num_layer += (net_config.block_repeat_size[block_idx] *
												net_config.block_param_size[block_idx])
			layer_idx = cum_num_layer - 1
			tf.logging.info("**** upsample layer id: %s **** "%(str(layer_idx)))
			upsampled_hid = upsample(net_config,
					hiddens[layer_idx], stride=stride, tgt_len=tgt_len)
			upsampled_hids.append(upsampled_hid)

		# add residual connection
		upsampled_hidden = upsampled_hids[-1]
		unpooled_hidden = upsampled_hids[0]
		if_skip_connetion = net_config.get('if_skip_connetion', True)
		if if_skip_connetion:
			tf.logging.info("**** apply if_skip_connetion **** ")
			output = upsampled_hidden + unpooled_hidden
		else:
			output = upsampled_hidden
			tf.logging.info("**** not apply if_skip_connetion **** ")

	return output, ret_dict

def bridge_deconv_layer(net_config, hiddens, input_mask,
								reuse=tf.AUTO_REUSE):
	net_config = net_config
	ret_dict = {}
	if net_config.get("tgt_len", None):
		tgt_len = net_config['tgt_len']
	else:
		tgt_len = tf.shape(input_mask)[1]
	with tf.variable_scope("upsampling_layer", reuse=reuse):
		features = hiddens[-1]
		input_shape = bert_utils.get_shape_list(features)
		# [kernel_width, output_depth, input_depth]
		filters = tf.get_variable(
						"upsample_filter",
						shape=[4, input_shape[-1], input_shape[-1]])
		output_shape = [input_shape[0], tgt_len, input_shape[-1]]
		block_size = sum(net_config.block_size.split("_"))
		strides = np.power(net_config.pooling_size, block_size-1)
		if check_tf_version():
			upsampled_hidden = tf.nn.conv1d_transpose(
										features,
										filters,
										output_shape,
										strides,
										padding='SAME')
			tf.logging.info(" using tf conv1d_transpose")
		else:
			upsampled_hidden = conv1d_transpose.conv1d_transpose(
										features,
										filters,
										output_shape,
										strides,
										padding='SAME')
			tf.logging.info(" using tf-out conv1d_transpose")
		unpooled_hidden = upsampled_hids[0]
		if_skip_connetion = net_config.get('if_skip_connetion', True)
		if if_skip_connetion:
			tf.logging.info("**** apply if_skip_connetion **** ")
			output = upsampled_hidden + unpooled_hidden
		else:
			output = upsampled_hidden
			tf.logging.info("**** not apply if_skip_connetion **** ")
		return output, ret_dict

def tfmxl_layer(net_config, q, k, v, pos_enc, seg_mat, attn_mask, 
								is_training,
								initializer,
                func_mask=None, attn_bias=None,
                name="tfmxl"):

  """Single transformer-xl layer."""
  net_config = net_config

  ret_dict = {}
  output, attn_dict = funnel_transformer_ops.rel_multihead_attn(
  		net_config=net_config,
      q=q,
      k=k,
      v=v,
      pos_enc=pos_enc,
      seg_mat=seg_mat,
      attn_mask=attn_mask,
      attn_bias=attn_bias,
      d_model=net_config.d_model,
      n_head=net_config.n_head,
      d_head=net_config.d_head,
      dropout=net_config.dropout,
      dropatt=net_config.dropatt,
      is_training=is_training,
      initializer=initializer,
      func_mask=func_mask,
      rel_attn_type=net_config.rel_attn_type,
      name=name)

  output, pffn_dict = funnel_transformer_ops.positionwise_ffn(
      inp=output,
      d_model=net_config.d_model,
      d_inner=net_config.d_inner,
      activation_type=net_config.ff_activation,
      dropout=net_config.dropout,
      dropact=net_config.dropact,
      is_training=is_training,
      initializer=initializer,
      name=name)

  funnel_transformer_ops.update_ret_dict(ret_dict, attn_dict, "attn")
  funnel_transformer_ops.update_ret_dict(ret_dict, pffn_dict, "pffn")
  return output, ret_dict