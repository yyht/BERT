odpscmd=$1
pai_command="
	pai -name tensorboard
		-DsummaryDir='oss://alg-misc/BERT/bert_pretrain/open_domain/pretrain_single_random_hard_gan/electra_debug_only_mask_generator_debug/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com'
"

echo "${pai_command}"
${odpscmd} -e "${pai_command}"