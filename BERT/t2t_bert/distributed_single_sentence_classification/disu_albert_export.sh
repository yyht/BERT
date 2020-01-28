CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export_api.py \
 	--buckets "/data/xuht" \
	--config_file "./data/roberta_zh_l12_albert/bert_config_tiny.json" \
	--init_checkpoint "disu_albert/model.ckpt-53400" \
	--vocab_file "sentence_embedding/data/char_id.txt" \
	--label_id "./data/lcqmc/label_dict.json" \
	--max_length 128 \
	--train_file "porn/clean_data/textcnn/distillation/train_tfrecords" \
	--dev_file "porn/clean_data/textcnn/distillation/test_tfrecords" \
	--model_output "disu_albert/" \
	--export_dir "disu_albert/export" \
	--epoch 8 \
	--num_classes 2 \
	--train_size 952213 \
	--eval_size 238054 \
	--batch_size 24 \
	--model_type "albert" \
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
	--input_target "" \
	--decay "no" \
	--warmup "no" \
	--distillation "normal" \
    --task_type "single_sentence_classification" \
    --ln_type "postln"

 


