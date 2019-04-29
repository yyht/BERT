python ./t2t_bert/distributed_data_prepare/multitask_classification_data_prepare.py \
	--buckets /data/xuht \
	--multitask_dict ./t2t_bert/distributed_multitask/multi_task.json \
	--vocab_file w2v/tencent_ai_lab/char_id.txt \
	--lower_case True \
	--max_length 128