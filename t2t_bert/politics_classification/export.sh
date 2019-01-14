CUDA_VISIBLE_DEVICES="" python export.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --model_dir "/data/xuht/politics/model/" \
 --label2id "/data/xuht/politics/label_dict.json" \
 --init_checkpoint "/data/xuht/politics/model/model.ckpt" \
 --max_length 128 \
 --export_path "/data/xuht/politics/export" \
 --export_type "2"