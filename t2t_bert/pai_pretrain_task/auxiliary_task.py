import sys, os
sys.path.append("..")

import tensorflow as tf
from task_module import pretrain

def masked_lm(model_config, model, features, reuse=None):
	masked_lm_positions = features["masked_lm_positions"]
	masked_lm_ids = features["masked_lm_ids"]
	masked_lm_weights = features["masked_lm_weights"]
	(masked_lm_loss,
	masked_lm_example_loss, 
	masked_lm_log_probs) = pretrain.get_masked_lm_output(
									model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(),
									masked_lm_positions, 
									masked_lm_ids, 
									masked_lm_weights,
									reuse=None)

	return {"masked_lm_loss":masked_lm_loss, 
			"masked_lm_example_loss":masked_lm_example_loss,
			"masked_lm_log_probs":masked_lm_log_probs,
			"scope":"cls/predictions"}

def next_sentence_prediction(model_config, model, features, reuse=None):
	next_sentence_labels = features["next_sentence_label"]

	(next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = pretrain.get_next_sentence_output(
         model_config, model.get_pooled_output(), next_sentence_labels, reuse=None)

     return {"next_sentence_loss":next_sentence_loss,
     		"next_sentence_example_loss":next_sentence_example_loss,
     		"next_sentence_log_probs":next_sentence_log_probs,
     		"scope":"cls/seq_relationship"}