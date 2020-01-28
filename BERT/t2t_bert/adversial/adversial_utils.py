from model.bert import bert
from model_io import model_io
# from optimizer import optimizer
from optimizer import hvd_distributed_optimizer as optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils
from model.regularizer import vib
from utils.attention import attention_utils

def get_perturbation(model_config, opt, 
					looked_up_repres, loss, tvars):
	perturb = None
	for var in tvars:
		if var.name.split("/")[-1] == "word_embeddings:0":
			raw_perturb = opt.compute_gradients(loss, [var])[0]
			break
		normalized_per = tf.nn.l2_normalize(raw_perturb, axis=[1, 2])
		perturb = model_config.alpha*tf.sqrt(tf.cast(tf.shape(looked_up_repres)[2], tf.float32)) * tf.stop_gradient(normalized_per)
		break
	return perturb