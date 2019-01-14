python export.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --model_dir "/data/xuht/porn/model/" \
 --label2id "/data/xuht/porn/label_dict.json" \
 --init_checkpoint "/data/xuht/porn/model/oqmrc_8.ckpt" \
 --max_length 128 \
 --export_path "/data/xuht/porn/export" \
 --export_type "2"