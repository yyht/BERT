CUDA_VISIBLE_DEVICES="0" mpirun -np 1 \
 -H localhost:4 \
 python official_wsdm.py \
 --eval_data_file "/data/xuht/wsdm19/data/test.csv" \
 --output_file "/data/xuht/wsdm19/postprocess/data_12_6/unidirection/test.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/wsdm19/postprocess/data_12_6/unidirection/model_12_6/oqmrc.ckpt" \
 --result_file "/data/xuht/wsdm19/postprocess/data_12_6/unidirection/model_12_6/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --lang "zh" \
 --max_length 128

