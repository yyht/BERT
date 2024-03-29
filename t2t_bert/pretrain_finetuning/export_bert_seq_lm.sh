CUDA_VISIBLE_DEVICES="" python ./t2t_bert/pretrain_finetuning/export_api.py \
 --buckets "/data/xuht" \
 --config_file "/data/xuht/chinese_gpt_tiny/bert_config_tiny.json" \
 --model_dir "chinese_gpt_tiny/export" \
 --init_checkpoint "chinese_gpt_tiny/model.ckpt-429680" \
 --model_output "chinese_gpt_tiny/model.ckpt-429680" \
 --max_length 128 \
 --export_dir "chinese_gpt_tiny/export/infer_inputs_128" \
 --num_classes 2 \
 --input_target "" \
 --model_scope "bert" \
 --model_type "bert_seq" \
 --task_type "bert_seq_lm"  \
 --export_model_type "bert_seq_lm" \
 --seq_type "seq2seq" \
 --mask_type "left2right"