CUDA_VISIBLE_DEVICES="" python ./t2t_bert/pretrain_finetuning/export_api.py \
 --buckets "/data/xuht" \
 --config_file "/data/xuht/electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/bert_config_tiny_large_embed.json" \
 --model_dir "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/export" \
 --init_checkpoint "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200" \
 --model_output "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200" \
 --max_length 256 \
 --export_dir "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/export" \
 --num_classes 2 \
 --input_target "" \
 --model_scope "bert" \
 --model_type "bert" \
 --export_type "generator" \
 --sharing_mode "all_sharing"

CUDA_VISIBLE_DEVICES="" python ./t2t_bert/pretrain_finetuning/export_api.py \
 --buckets "/data/xuht" \
 --config_file "/data/xuht/electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/bert_config_tiny_large_embed.json" \
 --model_dir "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/export" \
 --init_checkpoint "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/discriminator.ckpt-1074200" \
 --model_output "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/discriminator.ckpt-1074200" \
 --max_length 256 \
 --export_dir "electra/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/discriminator/export" \
 --num_classes 2 \
 --input_target "" \
 --model_scope "bert" \
 --model_type "bert" \
 --export_type "discriminator" \
 --sharing_mode "all_sharing"