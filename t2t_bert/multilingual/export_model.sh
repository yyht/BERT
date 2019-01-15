CUDA_VISIBLE_DEVICES="" python export_model.py \
 --config_file "/data/xuht/multi_cased_L-12_H-768_A-12/bert_config.json" \
 --model_dir "/data/xuht/lazada/20190107/model/" \
 --label2id "/data/xuht/lazada/label_dict.json" \
 --init_checkpoint "/data/xuht/lazada/20190107/model_1_15/model_0.ckpt" \
 --max_length 128 \
 --export_path "/data/xuht/lazada/20190107/export" \
 --export_type "2"