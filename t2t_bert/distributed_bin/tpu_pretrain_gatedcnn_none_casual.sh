nohup python ./t2t_bert/distributed_bin/tpu_train_eval_api.py \
	--buckets "gs://yyht_source/pretrain" \
	--config_file "./data/textcnn/textcnn_multilingual_embedding_light_dgcnn_v1_bi_v1.json" \
	--init_checkpoint "model/tiny/bert_tiny_with_single_random_adam_decay_40_mixed_mask/model.ckpt-1145800" \
	--vocab_file "./data/chinese_L-12_H-768_A-12/vocab.txt" \
	--label_id "./data/lcqmc/label_dict.json" \
	--max_length 256 \
	--train_file "mnli_pretrain/wiki_book_corpus.txt.256.tfrecords,mnli_pretrain/wiki_zh_pretrain.txt.256.tfrecords,mnli_pretrain/multinli.train.ar.tsv.256.tfrecords,mnli_pretrain/multinli.train.bg.tsv.256.tfrecords,mnli_pretrain/multinli.train.de.tsv.256.tfrecords,mnli_pretrain/multinli.train.el.tsv.256.tfrecords,mnli_pretrain/multinli.train.en.tsv.256.tfrecords,mnli_pretrain/multinli.train.es.tsv.256.tfrecords,mnli_pretrain/multinli.train.fr.tsv.256.tfrecords,mnli_pretrain/multinli.train.hi.tsv.256.tfrecords,mnli_pretrain/multinli.train.ru.tsv.256.tfrecords,mnli_pretrain/multinli.train.sw.tsv.256.tfrecords,mnli_pretrain/multinli.train.th.tsv.256.tfrecords,mnli_pretrain/multinli.train.tr.tsv.256.tfrecords,mnli_pretrain/multinli.train.ur.tsv.256.tfrecords,mnli_pretrain/multinli.train.vi.tsv.256.tfrecords,mnli_pretrain/multinli.train.zh.tsv.256.tfrecords,mnli_pretrain/train.json.256.tfrecords.not_selected,mnli_pretrain/train.json.256.tfrecords.selected" \
	--dev_file "data_single_hard_gan/chunk_18.tfrecords,data_single_hard_gan/chunk_19.tfrecords" \
	--model_output "model/gatedcnn_seq/gatedcnn_seq_light_v1_disc_bi_smaller" \
	--epoch 50 \
	--num_classes 2 \
	--train_size 6000000 \
	--eval_size 1100000 \
	--batch_size 256 \
	--model_type "gated_cnn_seq" \
	--model_scope "textcnn" \
	--if_shard 1 \
	--is_debug 1 \
	--profiler "no" \
	--train_op "adam_decay" \
	--load_pretrained "no" \
	--with_char "no_char" \
	--input_target "b" \
	--task_type "bert_pretrain" \
	--max_predictions_per_seq 78 \
	--ln_type "postln" \
	--warmup "warmup" \
	--decay "decay" \
	--init_lr 5e-4 \
	--do_train true \
	--num_tpu_cores 8 \
	--tpu_name "albert2" \
	--mode "pretrain" \
	--seq_type "seq2seq" \
	--mask_type "left2right" \
	--random_generator "4"




