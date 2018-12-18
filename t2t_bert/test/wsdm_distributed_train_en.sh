CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun -np 4 \
 -H localhost:4 \
 python test_wsdm_distributed.py \
 --eval_data_file "/data/xuht/wsdm19/test.csv" \
 --output_file "/data/xuht/wsdm19/order/dev.tfrecords" \
 --config_file "/data/xuht/bert/uncased_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
 --result_file "/data/xuht/wsdm19/order/model_11_15/submission.csv" \
 --vocab_file "/data/xuht/bert/uncased_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --train_file "/data/xuht/wsdm19/postprocess/data_12_6/unidirection_en/train.tfrecords" \
 --dev_file "/data/xuht/wsdm19/postprocess/data_12_6/unidirection_en/dev.tfrecords" \
 --max_length 128 \
 --model_output "/data/xuht/wsdm19/postprocess/data_12_6/unidirection_en/model_12_6" \
 --gpu_id "0" \
 --epoch 5 \
 --num_classes 3 \
 --model_type "original" \
 --batch_size 32

