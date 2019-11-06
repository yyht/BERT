nohup python ./t2t_bert/distributed_bin/tpu_train_eval_api.py \
	--buckets "gs://yyht_source/pretrain" \
	--config_file "./data/roberta_zh_l12_albert/bert_config_tiny.json" \
	--init_checkpoint "model/albert_tiny_factorized_with_single_adam_decay_dropout/model.ckpt-119000" \
	--vocab_file "./data/chinese_L-12_H-768_A-12/vocab.txt" \
	--label_id "./data/lcqmc/label_dict.json" \
	--max_length 512 \
	--train_file "data/chunk_0.tfrecords,data/chunk_1.tfrecords,data/chunk_2.tfrecords,data/chunk_3.tfrecords,data/chunk_4.tfrecords,data/chunk_5.tfrecords,data/chunk_6.tfrecords,data/chunk_7.tfrecords,data/chunk_8.tfrecords,data/chunk_9.tfrecords,data/chunk_10.tfrecords,data/chunk_11.tfrecords,data/chunk_12.tfrecords,data/chunk_13.tfrecords,data/chunk_14.tfrecords,data/chunk_15.tfrecords,data/chunk_16.tfrecords,data/chunk_17.tfrecords" \
	--dev_file "data_single/chunk_18.tfrecords,data_single/chunk_19.tfrecords" \
	--model_output "model/albert_tiny_factorized_with_single_adam_decay_dropout" \
	--epoch 15 \
	--num_classes 2 \
	--train_size 11000000 \
	--eval_size 1100000 \
	--batch_size 1024 \
	--model_type "albert" \
	--if_shard 1 \
	--is_debug 1 \
	--profiler "no" \
	--train_op "adam_decay" \
	--load_pretrained "yes" \
	--with_char "no_char" \
	--input_target "" \
	--task_type "bert_pretrain" \
	--max_predictions_per_seq 78 \
	--ln_type "postln" \
	--warmup "warmup" \
	--decay "decay" \
	--init_lr 1e-4 \
	--tpu_name "htxu91"



