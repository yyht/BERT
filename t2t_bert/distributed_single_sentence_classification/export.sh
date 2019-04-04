CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export_api.py \
 	--buckets "/data/xuht" \
	--config_file "./data/textlstm/textlstm.json" \
	--init_checkpoint "sentence_embedding/data/model/textlstm_0329/model.ckpt-517401" \
	--vocab_file "sentence_embedding/data/char_id.txt" \
	--label_id "/data/xuht/sentence_embedding/cluster_corpus_label_dict.json" \
	--max_length 128 \
	--train_file "porn/clean_data/textcnn/distillation/train_tfrecords" \
	--dev_file "porn/clean_data/textcnn/distillation/test_tfrecords" \
	--model_output "sentence_embedding/data/model/textlstm_0329/" \
	--export_dir "sentence_embedding/data/model/textlstm_0329/export" \
	--epoch 8 \
	--num_classes 5 \
	--train_size 952213 \
	--eval_size 238054 \
	--batch_size 24 \
	--model_type "textlstm" \
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
	--w2v_path "w2v/tencent_ai_lab/char_w2v.txt" \
	--with_char "no_char" \
	--input_target "a" \
	--decay "no" \
	--warmup "no" \
	--distillation "normal" \
    --task_type "embed_sentence_classification"

 


