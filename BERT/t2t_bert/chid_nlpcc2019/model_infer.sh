CUDA_VISIBLE_DEVICES="" python ./model_infer.py \
	--buckets "/data/xuht" \
	--input_file "nlpcc2019/chid/data/dev/dev_input.json" \
	--output_file "nlpcc2019/chid/data/dev/dev_output_max.json" \
	--model_file "nlpcc2019/chid/data/model/estimator/bert_chid_0816_ema/export/1566284155/" \
	--score_merge "max"