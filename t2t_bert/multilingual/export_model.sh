python export_model.py \
 --config_file "/data/xuht/multi_cased_L-12_H-768_A-12/bert_config.json" \
 --model_dir "/data/xuht/lazada/model/" \
 --label2id "/data/xuht/lazada/label_dict.json" \
 --init_checkpoint "/data/xuht/lazada/model/oqmrc_2.ckpt" \
 --max_length 128 \
 --export_path "/data/xuht/lazada/export" \
 --export_type "2"