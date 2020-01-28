mpirun -np 4 \
 -H localhost:4 \
 python test_oqmrc_distributed_final.py \
 --eval_data_file "/data/xuht/wsdm19/test.csv" \
 --output_file "/data/xuht/wsdm19/interaction/dev.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --result_file "/data/xuht/wsdm19/interaction/model_11_15/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --train_file "/data/xuht/oqmrc/concat/data/train.tfrecords" \
 --dev_file "/data/xuht/oqmrc/concat/data/test.tfrecords" \
 --max_length 200 \
 --model_output "/data/xuht/oqmrc/concat/data/model_12_2_distributed" \
 --gpu_id "0" \
 --epoch 5 \
 --num_classes 3 \
 --batch_size 8

