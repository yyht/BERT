CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 \
 -H localhost:2 \
python base_train.py \
 --config_file "/data/xuht/multi_cased_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/multi_cased_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/multi_cased_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/lazada/label_dict.json" \
 --train_file "/data/xuht/lazada/20190107/train_tf_records" \
 --dev_file "/data/xuht/lazada/20190107/test_tf_records" \
 --max_length 128 \
 --model_output "/data/xuht/lazada/20190107/model_1_15" \
 --epoch 5 \
 --num_classes 4 \
 --batch_size 32 \
 --if_shard "1" \

 
