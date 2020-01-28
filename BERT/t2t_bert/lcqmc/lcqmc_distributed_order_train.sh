CUDA_VISIBLE_DEVICES="2,3" mpirun -np 2 \
 -H localhost:4 \
 python test_lcqmc_distributed_order.py \
 --eval_data_file "/data/xuht/wsdm19/test.csv" \
 --output_file "/data/xuht/wsdm19/order/dev.tfrecords" \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --result_file "/data/xuht/wsdm19/order/model_11_15/submission.csv" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/LCQMC/label_dict.json" \
 --train_file "/data/xuht/LCQMC/train.tfrecords" \
 --dev_file "/data/xuht/LCQMC/dev.tfrecords" \
 --max_length 128 \
 --model_output "/data/xuht/LCQMC/model/model_12_5" \
 --gpu_id "0" \
 --epoch 5 \
 --num_classes 2 \
 --model_type "original"

