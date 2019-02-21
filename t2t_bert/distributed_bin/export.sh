CUDA_VISIBLE_DEVICES="" python export.py \
 --buckets "/data/xuht" \
 --local_buckets "/data/xuht" \
 --config_file "chinese_L-12_H-768_A-12/bert_config.json" \
 --model_dir "jd_comment/hvd/rule_data/model_sess/" \
 --label2id "jd_comment/label_dict.json" \
 --init_checkpoint "jd_comment/hvd/rule_data/model_sess/model-1032.ckpt" \
 --max_length 128 \
 --export_path "jd_comment/hvd/rule_data/model_sess/export" \
 --export_type "2"