import tensorflow as tf
import numpy as np

try:
	from .single_sentence_bert_teacher import teacher_model_fn_builder
	from .single_sentence_cnn_student import student_model_fn_builder
except:
	from single_sentence_bert_teacher import model_fn_builder as teacher_model_fn_builder
	from single_sentence_cnn_student import model_fn_builder as student_model_fn_builder

