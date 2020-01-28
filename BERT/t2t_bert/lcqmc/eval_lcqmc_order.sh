CUDA_VISIBLE_DEVICES="2" python eval_lcqmc_order.py \
 --eval_data_file "/data/xuht/wsdm19/data/test.csv" \
 --output_file "/data/xuht/LCQMC/test.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/LCQMC/model/model_12_5/oqmrc_3.ckpt" \
 --result_file "/data/xuht/wsdm19/interaction/model_11_21/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/LCQMC/label_dict.json" \
 --max_length 128 \
 --model_type "original"
