# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
# 	--checkpoint_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_1_grl_only_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_1_grl_only_mask/discriminator/discriminator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
# 	--checkpoint_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_1_grl_only_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_1_grl_only_mask/generator/generator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra_shard_embedding/model.ckpt-0' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra_shard_embedding/discriminator/discriminator.ckpt-0'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra_shard_embedding/model.ckpt-0' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra_shard_embedding/generator/generator.ckpt-0'


# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra/discriminator/bert_config.json' \
# 	--checkpoint_path 'electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4/model.ckpt-1289000' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4/discriminator/discriminator.ckpt-1289000'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra/generator/bert_config_small.json' \
# 	--checkpoint_path 'electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4/model.ckpt-1289000' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4/generator/generator.ckpt-1289000'

python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
	--buckets '/data/xuht' \
	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_large_embed.json' \
	--checkpoint_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding/model.ckpt-1070000' \
	--model_type 'bert' \
	--electra 'discriminator' \
	--model_scope 'bert' \
	--exclude_scope '' \
	--export_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding/discriminator/discriminator.ckpt-1070000'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding/model.ckpt-1070000' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding/generator/generator.ckpt-1070000'
