python ./t2t_bert/dataset_generator/write_to_tfrecords.py \
    --buckets "/data/xuht" \
	--multi_task_config "./t2t_bert/distributed_multitask/multi_task_local.json" \
	--batch_size 32 \
	--model_output "multi_task/data_generator" \
	--multi_task_type "wsdm,ccks,ant,lcqmc,chnsenticorp,xnli,nlpcc-dbqa" \
	--epoch 10