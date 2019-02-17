mpirun -np 2 \
 -H localhost:2 \
 python ./t2t_bert/distributed_bin/hvd_train_eval_api.py \
 --buckets "/data/xuht" \
 --config_file "bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "bert/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "jd_comment/label_dict.json" \
 --train_file "jd_comment/train.tfrecords" \
 --dev_file "jd_comment/test.tfrecords" \
 --max_length 128 \
 --model_output "jd_comment/hvd/model" \
 --epoch 2 \
 --num_classes 2 \
 --batch_size 32 \
 --train_size 33033 \
 --eval_size 8589 \
 --model_type bert \
 --if_shard "1" \
 --is_debug "1" \
 --run_type "sess" \
 --opt_type "hvd"

