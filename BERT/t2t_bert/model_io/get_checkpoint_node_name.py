from tensorflow.python import pywrap_tensorflow
checkpoint_path = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/model.ckpt-2841'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
	print("tensor_name: ", key)