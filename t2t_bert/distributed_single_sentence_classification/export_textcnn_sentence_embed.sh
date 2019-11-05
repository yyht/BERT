CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export_api.py \
 	--buckets "/data/xuht" \
	--config_file "./data/sentence_embedding/textcnn/textcnn.json" \
	--init_checkpoint "sentence_embedding/graph/bert_embed_data/model/textcnn_0421/model.ckpt-71721" \
	--vocab_file "chinese_L-12_H-768_A-12/vocab.txt" \
	--label_id "/data/xuht/sentence_embedding/graph/cluster_corpus_label_dict.json" \
	--max_length 128 \
	--train_file "porn/clean_data/textcnn/distillation/train_tfrecords" \
	--dev_file "porn/clean_data/textcnn/distillation/test_tfrecords" \
	--model_output "sentence_embedding/graph/bert_embed_data/model/textcnn_0421" \
	--export_dir "sentence_embedding/graph/bert_embed_data/model/textcnn_0421/export" \
	--epoch 8 \
	--num_classes 5 \
	--train_size 952213 \
	--eval_size 238054 \
	--batch_size 24 \
	--model_type "textcnn" \
	--if_shard 2 \
	--is_debug 1 \
	--run_type "sess" \
	--opt_type "all_reduce" \
	--num_gpus 4 \
	--parse_type "parse_batch" \
	--rule_model "normal" \
	--profiler "no" \
	--train_op "adam_weight_decay_exclude" \
	--running_type "eval" \
	--cross_tower_ops_type "paisoar" \
	--distribution_strategy "MirroredStrategy" \
	--load_pretrained "no" \
	--w2v_path "chinese_L-12_H-768_A-12/vocab_w2v.txt" \
	--with_char "no_char" \
	--input_target "a" \
	--decay "no" \
	--warmup "no" \
	--distillation "normal" \
    --task_type "embed_sentence_classification"

 


