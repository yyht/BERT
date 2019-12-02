CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export_api.py \
 	--buckets "/data/xuht" \
	--config_file "./data/textcnn_feature_distillation/textcnn.json" \
	--init_checkpoint "lazada/new_data/20190521/data/distillation/st/model/textcnn_0614_128_no_lower/model.ckpt-403393" \
	--vocab_file "multi_cased_L-12_H-768_A-12/vocab.txt" \
	--label_id "./data/lazada_multilingual/label_dict.json" \
	--max_length 128 \
	--train_file "porn/clean_data/textcnn/distillation/train_tfrecords" \
	--dev_file "porn/clean_data/textcnn/distillation/test_tfrecords" \
	--model_output "lazada/new_data/20190521/data/distillation/st/model/textcnn_0614_128_no_lower" \
	--export_dir "lazada/new_data/20190521/data/distillation/st/model/textcnn_0614_128_no_lower/export" \
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
	--w2v_path "multi_cased_L-12_H-768_A-12/vocab_w2v.txt" \
	--with_char "no_char" \
	--input_target "a" \
	--decay "no" \
	--warmup "no" \
	--distillation "normal" \
    --task_type "single_sentence_classification"

 


