CUDA_VISIBLE_DEVICES="0" python ps_train_eval.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/politics/label_dict.json" \
 --train_file "/data/xuht/politics/train_tf_records" \
 --dev_file "/data/xuht/politics/test_tf_records" \
 --max_length 128 \
 --model_output "/data/xuht/politics/model" \
 --epoch 2 \
 --num_classes 2 \
 --batch_size 32 \
 --if_shard "1" \

 
