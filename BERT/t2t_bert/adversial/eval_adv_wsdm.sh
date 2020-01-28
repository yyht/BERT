CUDA_VISIBLE_DEVICES="2,3" mpirun -np 1 \
 -H localhost:4 \
 python eval_adv_wsdm.py \
 --eval_data_file "/data/xuht/wsdm19/test.csv" \
 --output_file "/data/xuht/wsdm19/order/dev.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/wsdm19/data_12_3/adv/model_12_6/oqmrc_5.ckpt" \
 --result_file "/data/xuht/wsdm19/order/model_11_15/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --train_file "/data/xuht/wsdm19/data_12_3/order/train.tfrecords" \
 --dev_file "/data/xuht/wsdm19/data_12_3/order/dev.tfrecords" \
 --max_length 128 \
 --model_output "/data/xuht/wsdm19/data_12_3/adv/model_12_5" \
 --gpu_id "0" \
 --epoch 5 \
 --num_classes 3 \
 --model_type "original"

