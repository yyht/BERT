python ./t2t_bert/distributed_data_prepare/multitask_classification_data_prepare.py \
	--buckets /data/xuht \
	--multitask_dict multi_task/data/multi_task.json \
	--vocab_file ./data/chinese_L-12_H-768_A-12/vocab.txt \
	--lower_case True \
	--max_length 128 \
	--multi_task_type "wsdm,ccks,ant,xnli,lcqmc,nlpcc-dbqa,chnsenticorp"