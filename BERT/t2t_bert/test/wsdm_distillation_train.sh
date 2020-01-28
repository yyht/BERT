python test_wsdm_distillation.py \
 --eval_data_file "/data/xuht/wsdm19/test.csv" \
 --output_file "/data/xuht/wsdm19/order/dev.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --teacher_init_checkpoint "/data/xuht/wsdm19/order/model_11_21/oqmrc.ckpt" \
 --student_init_checkpoint "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --result_file "/data/xuht/wsdm19/order/model_11_15/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --train_file "/data/xuht/wsdm19/order/train.tfrecords" \
 --dev_file "/data/xuht/wsdm19/order/dev.tfrecords" \
 --max_length 100 \
 --model_output "/data/xuht/wsdm19/order/distillation/model_11_24" \
 --gpu_id "3" \
 --epoch 5 \
 --num_classes 3 \
 --batch_size 16

