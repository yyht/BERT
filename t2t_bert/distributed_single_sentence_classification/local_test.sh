CUDA_VISIBLE_DEVICES="0" python local_test.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/jd_comment/label_dict.json" \
 --train_file "/data/xuht/jd_comment/train.tfrecords" \
 --dev_file "/data/xuht/jd_comment/test.tfrecords" \
 --max_length 128 \
 --model_output "/data/xuht/jd_comment/model/ps_test" \
 --epoch 2 \
 --num_classes 2 \
 --batch_size 32 \
 --if_shard "1" \
 --opt_type "" \
 --is_debug "1" \
 --train_size 33033 \
 --eval_size 8589 \
 --run_type "sess"

 
