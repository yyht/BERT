import tensorflow as tf
import numpy as np

try:
	from .student_model_fn import model_fn_builder as st_model_fn
	from .teacher_model_fn import model_fn_builder as ta_model_fn
except:
	from student_model_fn import model_fn_builder as st_model_fn
	from teacher_model_fn import model_fn_builder as ta_model_fn

from model_io import model_io
from optimizer import distributed_optimizer as optimizer
from distillation import distillation_utils 
from distillation import flip_gradient
from distillation import mdd_utils
from distillation import repo_distillation_utils
from metric import tf_metrics
from distillation import uniform_mapping
from distillation import cpc_utils

def distillation_model_fn(model_config_dict,
					num_labels_dict,
					init_checkpoint_dict,
					load_pretrained_dict,
					model_io_config={},
					opt_config={},
					exclude_scope_dict={},
					not_storage_params_dict={},
					target_dict={},
					output_type="sess",
					distillation_config={},
					**kargs):

	def model_fn(features, labels, mode):

		original_loss = tf.constant(0.0)
		distilled_loss = tf.constant(0.0)

		st_model = st_model_fn(model_config_dict['student'],
		 			num_labels_dict['student'],
					init_checkpoint_dict['student'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['student'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('student', ""),
					not_storage_params=not_storage_params_dict.get('student', []),
					target=target_dict['student'],
					**kargs)
		st_dict = st_model(features, labels, mode)

		ta_model = ta_model_fn(model_config_dict['teacher'],
		 			num_labels_dict['teacher'],
					init_checkpoint_dict['teacher'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['teacher'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('teacher', ""),
					not_storage_params=not_storage_params_dict.get('teacher', []),
					target=target_dict['teacher'],
					**kargs)
		ta_dict = ta_model(features, labels, mode)

		studnet_logit = st_dict['logits']
		teacher_logit = ta_dict['logits']

		model_io_fn = model_io.ModelIO(model_io_config)

		feature_flag = False

		original_loss += st_dict['loss'] * (distillation_config.get('ce_loss', 1.0))
		print(distillation_config.get('ce_loss', 1.0), '===ce_loss===')

		