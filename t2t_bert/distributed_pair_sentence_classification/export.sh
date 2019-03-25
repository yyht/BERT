CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_bin/export_api.py \
 --buckets "/data/xuht" \
 --config_file "./data/match_pyramid/match_pyramid.json" \
 --init_checkpoint "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/restore/model.ckpt-186501" \
 --vocab_file "lcqmc/data/distillation/char_id.txt" \
 --label_id "/data/xuht/lcqmc/data/label_dict.json" \
 --max_length 64 \
 --train_file "lcqmc/data/distillation/train_tfrecords" \
 --dev_file "lcqmc/data/distillation/dev_tfrecords" \
 --model_output "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/restore" \
 --export_dir "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/export" \
 --epoch 50 \
 --num_classes 2 \
 --train_size 238766 \
 --eval_size 8802 \
 --batch_size 32 \
 --model_type "match_pyramid" \
 --if_shard 1 \
 --is_debug 1 \
 --run_type "sess" \
 --opt_type "hvd" \
 --num_gpus 4 \
 --parse_type "parse_batch" \
 --rule_model "normal" \
 --profiler "no" \
 --train_op "adam" \
 --running_type "train" \
 --cross_tower_ops_type "paisoar" \
 --distribution_strategy "MirroredStrategy" \
 --load_pretrained "yes" \
 --w2v_path "w2v/tencent_ai_lab/char_w2v.txt" \
 --with_char "no_char" \
 --input_target "a,b" \
 --decay "no" \
 --warmup "no" \
 --distillation "normal" \
 --temperature 2.0 \
 --distillation_ratio 0.5 \
 --task_type "interaction_pair_sentence_classification" \
 --classifier "siamese_interaction_classifier" \
 --output_layer "interaction"


