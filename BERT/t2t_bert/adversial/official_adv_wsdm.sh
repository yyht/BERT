CUDA_VISIBLE_DEVICES="2" python official_adv_wsdm.py \
 --eval_data_file "/data/xuht/wsdm19/data/test.csv" \
 --output_file "/data/xuht/wsdm19/data_12_3/adv/test.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/wsdm19/data_12_3/adv/model_12_6/oqmrc_5.ckpt" \
 --result_file "/data/xuht/wsdm19/data_12_3/adv/model_12_5/submision_5.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --lang "zh" \
 --max_length 128 \
 --model_type "original"

