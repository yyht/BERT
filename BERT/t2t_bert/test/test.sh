python official_oqmrc_test.py \
 --eval_data_file "/data/xuht/ai_challenge_oqmrc/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json" \
 --output_file "/data/xuht/concat/testa.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/concat/finial_finial_model/oqmrc_4.ckpt" \
 --result_file "/data/xuht/concat/result.txt" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt"

 