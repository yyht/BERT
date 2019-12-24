nohup python ./t2t_bert/glue_benchmark/crmc2018/run_baseline.py \
	--buckets "gs://yyht_source/pretrain" \
	--bert_config_file "./data/roberta_zh_l12_albert/bert_config.json" \
	--init_checkpoint "model/bert_base_with_single_random_generator_adam_decay_15/model.ckpt-644000" \
	--vocab_file "./data/chinese_L-12_H-768_A-12/vocab.txt" \
	--max_seq_length 512 \
	--max_query_length 64 \
	--doc_stride 128 \
	--warmup_proportion 0.1 \
	--train_file "chinese_glue/crmc_2018/cmrc2018_train.json" \
	--eval_file "chinese_glue/crmc_2018/cmrc2018_dev.json" \
	--output_dir "chinese_glue/crmc_2018_bert_base_dynamic_mask_adam_decay_15" \
	--epoch 5 \
	--train_size 10000 \
	--eval_size 3200 \
	--train_batch_size 64 \
	--predict_batch_size 64 \
	--model_type "bert" \
	--optimizer_type "tpu_adamw" \
	--ln_type "postln" \
	--learning_rate 3e-5 \
	--num_tpu_cores 8 \
	--tpu_name "albert1" \
	--mode "pretrain" \
	--exclude_scope "" \
	--ues_token_type "yes" \
	--do_train true
