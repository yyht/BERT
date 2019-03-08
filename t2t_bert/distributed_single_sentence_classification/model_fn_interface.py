try:
	from .model_fn import model_fn_builder
	from .model_distillation_fn import model_fn_builder as model_distillation_builder_fn
except:
	from model_fn import model_fn_builder
	from model_distillation_fn import model_fn_builder as model_distillation_builder_fn

def model_fn_interface(FLAGS):
	print("==apply {} model fn builder==".format(FLAGS.distillation))
	if FLAGS.distillation == "distillation":
		return model_distillation_builder_fn
	elif FLAGS.distillation == "normal":

		return model_fn_builder
	else:
		return model_fn_builder