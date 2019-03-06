mpirun -np 2 \
 -H localhost:2 \
 python ./t2t_bert/distributed_bin/hvd_train_eval_api.py \
 --buckets "/data/xuht" \
 --config_file "./data/textcnn/textcnn.json" \
 --init_checkpoint "" \
 --vocab_file "w2v/tencent_ai_lab/char_id.txt" \
 --label_id "porn/label_dict.json" \
 --max_length 128 \
 --train_file "porn/clean_data/textcnn/train_tfrecords" \
 --dev_file "porn/clean_data/textcnn/dev_tfrecords" \
 --model_output "porn/clean_data/textcnn/model/estimator/all_reduce_4_adam_weight_0228/" \
 --epoch 20 \
 --num_classes 5 \
 --train_size 952213 \
 --eval_size 238054 \
 --batch_size 24 \
 --model_type "textcnn" \
 --if_shard 1 \
 --is_debug 1 \
 --run_type "estimator" \
 --opt_type "all_reduce" \
 --num_gpus 4 \
 --parse_type "parse_batch" \
 --rule_model "normal" \
 --profiler "no" \
 --train_op "adam" \
 --running_type "train" \
 --cross_tower_ops_type "paisoar" \
 --distribution_strategy "MirroredStrategy" \
 --load_pretrained "no" \
 --w2v_path "w2v/tencent_ai_lab/char_w2v.txt" \
 --with_char "no_char" \
 --input_target "a" \
 --distribution_strategy "MirroredStrategy"


