mpirun -np 2 \
 -H localhost:2 \
 python ./t2t_bert/distributed_bin/hvd_train_eval_api.py \
 --buckets "/data/xuht" \
 --config_file "chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "porn/label_dict.json" \
 --train_file "porn/rule/seqing_train_20180821_tf_records" \
 --dev_file "porn/rule/seqing_test_20180821_tf_records" \
 --max_length 128 \
 --model_output "porn/rule/model/sess" \
 --epoch 2 \
 --num_classes 5 \
 --batch_size 32 \
 --train_size 1402171 \
 --eval_size 145019 \
 --model_type bert \
 --if_shard "1" \
 --is_debug "1" \
 --run_type "sess" \
 --opt_type "hvd" \
 --distribution_strategy "ParameterServerStrategy" \
 --rule_model "rule" \
 --parse_type "parse_batch" \
 --profiler "no" \
 --running_type "train" \
 --load_pretrained "no"


