nohup python ./t2t_bert/distributed_bin/tpu_train_eval_api.py \
	--buckets "gs://yyht_source/pretrain" \
	--config_file "./data/roberta_zh_l12_albert/bert_config_tiny.json" \
	--init_checkpoint "" \
	--vocab_file "./data/chinese_L-12_H-768_A-12/vocab.txt" \
	--label_id "./data/lcqmc/label_dict.json" \
	--max_length 512 \
	--train_file "new_data/chunk_0.tfrecords,new_data/chunk_1.tfrecords,new_data/chunk_2.tfrecords,new_data/chunk_3.tfrecords,new_data/chunk_4.tfrecords,new_data/chunk_5.tfrecords,new_data/chunk_6.tfrecords,new_data/chunk_7.tfrecords,new_data/chunk_8.tfrecords,new_data/chunk_9.tfrecords,new_data/chunk_10.tfrecords,new_data/chunk_11.tfrecords,new_data/chunk_12.tfrecords,new_data/chunk_13.tfrecords,new_data/chunk_14.tfrecords,new_data/chunk_15.tfrecords,new_data/chunk_16.tfrecords,new_data/chunk_17.tfrecords" \
	--dev_file "new_data/chunk_18.tfrecords,new_data/chunk_19.tfrecords" \
	--model_output "model/albert_tiny_factorized" \
	--epoch 50 \
	--num_classes 2 \
	--train_size 12000000 \
	--eval_size 1200000 \
	--batch_size 4096 \
	--model_type "albert" \
	--if_shard 1 \
	--is_debug 1 \
	--profiler "no" \
	--train_op "lamb_v2" \
	--load_pretrained "no" \
	--with_char "no_char" \
	--input_target "" \
	--task_type "bert_pretrain" \
	--max_predictions_per_seq 78 \
	--ln_type "postln" \
	--warmup "warmup" \
	--decay "decay" \
	--init_lr 1e-4 \
	--do_train true \
	--tpu_name "htxu91"



