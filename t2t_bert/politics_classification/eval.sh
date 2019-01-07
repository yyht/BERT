CUDA_VISIBLE_DEVICES="0" mpirun -np 1 \
 -H localhost:2 \
python eval.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/politics/model/model.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/politics/label_dict.json" \
 --train_file "/data/xuht/politics/global_mining/mining_info.pkl" \
 --dev_file "/data/xuht/politics/test_tf_records" \
 --max_length 64 \
 --model_output "/data/xuht/politics/global_mining/" \
 --epoch 5 \
 --num_classes 5 \
 --batch_size 32 \
 --if_shard "1" \
 --lower_case True \
 --eval_data_file "/data/xuht/politics/global_mining/eval.tfrecords"

