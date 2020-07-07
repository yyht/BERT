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

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/discriminator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200'


python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
	--buckets '/data/xuht' \
	--config_path './data/roberta_zh_l12/bert_config_tiny.json' \
	--checkpoint_path 'chinese_electra_tiny_green/model.ckpt-500000' \
	--model_type 'bert' \
	--electra 'discriminator' \
	--model_scope 'electra' \
	--exclude_scope '' \
	--export_path 'chinese_electra_tiny_green/discriminator/discriminator.ckpt-500000'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_embed_sharing.json' \
# 	--checkpoint_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/grl/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_embed_sharing.json' \
# 	--checkpoint_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_all/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_all/discriminator/discriminator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_embed_sharing.json' \
# 	--checkpoint_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_all/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_all/generator/generator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_embed_sharing.json' \
# 	--checkpoint_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_not_equal/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'discriminator' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_not_equal/discriminator/discriminator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_embed_sharing.json' \
# 	--checkpoint_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_not_equal/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/alternate/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_sharing_pretrained_embedding_mixed_mask_not_equal/generator/generator.ckpt-1074200'



# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/discriminator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra/relgan/soft/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'discriminator_relgan' \
# 	--model_scope 'bert' \
# 	--exclude_scope '' \
# 	--export_path 'electra/relgan/soft/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/discriminator.ckpt-1074200'

# python ./t2t_bert/pretrain_finetuning/export_checkpoints.py \
# 	--buckets '/data/xuht' \
# 	--config_path './data/electra_share_embedding/generator/bert_config_tiny_large_embed.json' \
# 	--checkpoint_path 'electra/relgan/soft/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/model.ckpt-1074200' \
# 	--model_type 'bert' \
# 	--electra 'generator' \
# 	--model_scope 'bert' \
# 	--exclude_scope 'generator' \
# 	--export_path 'electra/relgan/soft/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200'
