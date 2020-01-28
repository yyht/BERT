CUDA_VISIBLE_DEVICES="0" python official_wsdm_order.py \
 --eval_data_file "/data/xuht/wsdm19/data/test.csv" \
 --output_file "/data/xuht/wsdm19/postprocess/data_12_6/order/test.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/wsdm19/postprocess/data_12_6/order/model_12_5/oqmrc.ckpt" \
 --result_file "/data/xuht/wsdm19/postprocess/data_12_6/order/model_12_5/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --lang "zh" \
 --max_length 128 \
 --model_type "original"

