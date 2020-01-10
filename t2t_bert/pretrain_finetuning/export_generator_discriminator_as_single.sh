python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
	--buckets '/data/xuht' \
	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
	--checkpoint_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix_scratch/model.ckpt-576000' \
	--model_type 'bert' \
	--electra 'discriminator' \
	--model_scope 'bert' \
	--exclude_scope '' \
	--export_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix_scratch/discriminator/discriminator'

python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
	--buckets '/data/xuht' \
	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
	--checkpoint_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix_scratch/model.ckpt-576000' \
	--model_type 'bert' \
	--electra 'generator' \
	--model_scope 'bert' \
	--exclude_scope 'generator' \
	--export_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix_scratch/generator/generator'