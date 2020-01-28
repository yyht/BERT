CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export.py \
 --buckets "/data/xuht" \
 --local_buckets "/data/xuht" \
 --config_file "chinese_L-12_H-768_A-12/bert_config.json" \
 --model_dir "porn/rule/model/estimator/all_reduce_4_adam_weight/" \
 --label_id "porn/label_dict.json" \
 --init_checkpoint "porn/rule/model/estimator/all_reduce_4_adam_weight/model.ckpt-116840" \
 --max_length 128 \
 --export_path "porn/rule/model/estimator/all_reduce_4_adam_weight/export" \
 --export_type "2"