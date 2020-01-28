python test_tf_serving.py \
	--vocab "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
	--url "10.183.20.12" \
	--port "7901" \
	--model_name "bert" \
	--input_keys "instances" \
	--signature_name "serving_default" \
	--versions "1548673483"
