model_name=$1
model_base_path=$2

CUDA_VISIBLE_DEVICES="" tensorflow_model_server \
	--port=8500 \
	--rest_api_port=8501 \
	--model_name=${model_name}  \
	--model_base_path=${model_base_path}