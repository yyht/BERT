python official_oqmrc_test.py \
 --eval_data_file "data/ai_challenger_oqmrc_testa.json" \
 --output_file "data/testa.tfrecords" \
 --config_file "data/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "data/final_model/oqmrc_2.ckpt" \
 --result_file "data/result.txt" \
 --vocab_file "data/chinese_L-12_H-768_A-12/vocab.txt"

 