nohup python ./t2t_bert/distributed_bin/tpu_train_eval_api.py \
	--buckets "gs://yyht_source/pretrain" \
	--config_file "./data/multi_cased_L-12_H-768_A-12/bert_config_tiny_embedding.json" \
	--init_checkpoint "" \
	--vocab_file "./data/uncased_L-12_H-768_A-12/vocab.txt" \
	--label_id "./data/lcqmc/label_dict.json" \
	--max_length 128 \
	--train_file "mnli_xlm_tiny/multinli.train.ar.tsv.tfrecords,mnli_xlm_tiny/multinli.train.bg.tsv.tfrecords,mnli_xlm_tiny/multinli.train.de.tsv.tfrecords,mnli_xlm_tiny/multinli.train.el.tsv.tfrecords,mnli_xlm_tiny/multinli.train.en.tsv.tfrecords,mnli_xlm_tiny/multinli.train.es.tsv.tfrecords,mnli_xlm_tiny/multinli.train.fr.tsv.tfrecords,mnli_xlm_tiny/multinli.train.hi.tsv.tfrecords,mnli_xlm_tiny/multinli.train.ru.tsv.tfrecords,mnli_xlm_tiny/multinli.train.sw.tsv.tfrecords,mnli_xlm_tiny/multinli.train.th.tsv.tfrecords,mnli_xlm_tiny/multinli.train.tr.tsv.tfrecords,mnli_xlm_tiny/multinli.train.ur.tsv.tfrecords,mnli_xlm_tiny/multinli.train.vi.tsv.tfrecords,mnli_xlm_tiny/multinli.train.zh.tsv.tfrecords,mnli_xlm_tiny/de_new_train.tfrecords,mnli_xlm_tiny/en_new_train.tfrecords,mnli_xlm_tiny/es_new_train.tfrecords,mnli_xlm_tiny/fr_new_train.tfrecords,mnli_xlm_tiny/ja_new_train.tfrecords,mnli_xlm_tiny/ko_new_train.tfrecords,mnli_xlm_tiny/zh_new_train.tfrecords" \
	--dev_file "english_corpus/pretrain_single_random_gan_uncased/chunk_18.tfrecords,english_corpus/pretrain_single_random_gan_uncased/chunk_19.tfrecords" \
	--model_output "model/tiny/xlm/bert_tiny_with_single_random_adam_decay_40_mixed_mask_uncased" \
	--epoch 40 \
	--num_classes 2 \
	--train_size 11000000 \
	--eval_size 1100000 \
	--batch_size 512 \
	--model_type "bert" \
	--if_shard 1 \
	--is_debug 1 \
	--profiler "no" \
	--train_op "adam_decay" \
	--load_pretrained "no" \
	--with_char "no_char" \
	--input_target "" \
	--task_type "bert_pretrain" \
	--max_predictions_per_seq 78 \
	--ln_type "postln" \
	--warmup "warmup" \
	--decay "decay" \
	--init_lr 2e-4 \
	--num_tpu_cores 8 \
	--do_train true \
	--tpu_name "albert0" \
	--mode "pretrain" \
	--random_generator "3"




