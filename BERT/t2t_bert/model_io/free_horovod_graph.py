from tensorflow.python.tools import freeze_graph
import horovod.tensorflow
from tensorflow.python.saved_model import tag_constants

input_graph_path = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/graph.pbtxt'
checkpoint_path = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/model.ckpt-2841'
input_saver_def_path = ""
input_binary = False
output_node_names = "Sigmoid:0"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/data_security_frozen.pb'
output_optimized_graph_name = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/data_security_optimized.pb'
clear_devices = True
input_saved_model_dir = '/data/xuht/data_security/model/textcnn/model/20190725_16/textcnn_20190725_16/export/1564037890'
saved_model_tags = tag_constants.SERVING

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
						  input_binary, checkpoint_path, output_node_names,
						  restore_op_name, filename_tensor_name,
						  output_frozen_graph_name, clear_devices, "",
						  input_saved_model_dir=input_saved_model_dir,
						  saved_model_tags=saved_model_tags
						  )