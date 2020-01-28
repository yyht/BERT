
from distillation import distillation_utils
 
def daan_distillation(source, target,
					source_label, target_label,
					dropout_prob, num_labels)
	with tf.variable_scope(self.config.get("scope", "bert")+"/dann_distillation", reuse=tf.AUTO_REUSE):
		source_proj = tf.layers.dense(source,
								source.get_shape()[-1],
								activation=tf.nn.tanh,
								name="shared_encoder")
		[src_loss, 
		src_example_loss, 
		src_logits] = distillation_utils.feature_distillation(source_proj, 1.0, 
										source_label, num_labels,
										dropout_prob,
										if_gradient_flip=True)

	with tf.variable_scope(self.config.get("scope", "bert")+"/dann_distillation", reuse=tf.AUTO_REUSE):

		target_proj = tf.layers.dense(target,
								source.get_shape()[-1],
								activation=tf.nn.tanh,
								name="shared_encoder")

		[tgt_loss, 
		tgt_example_loss, 
		tgt_logits] = distillation_utils.feature_distillation(target_proj, 1.0, 
										target_label, num_labels,
										dropout_prob,
										if_gradient_flip=True)

		distillation_feature_loss = (tgt_loss + src_loss)
	return (distillation_feature_loss, src_example_loss, tgt_example_loss,
			src_logits,tgt_logits)