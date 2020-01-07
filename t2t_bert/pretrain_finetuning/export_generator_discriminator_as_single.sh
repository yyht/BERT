python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
	--buckets '/data/xuht' \
	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
	--checkpoint_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_none_sharing_adam_decay_alternate_real/model.ckpt-644520' \
	--model_type 'bert' \
	--electra 'discriminator' \
	--model_scope 'bert' \
	--exclude_scope '' \
	--export_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_none_sharing_adam_decay_alternate_real/discriminator/discriminator'