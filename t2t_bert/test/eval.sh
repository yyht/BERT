python eval_oqmrc_test.py \
 --eval_data_file "/data/xuht/concat/ai_challenger_oqmrc_testa.json" \
 --output_file "/data/xuht/oqmrc/concat/data/test.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/oqmrc/concat/data/model_12_2_distributed/oqmrc_epoch_3_hvd_3.ckpt" \
 --result_file "/data/xuht/concat/data/result.txt" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt"

 