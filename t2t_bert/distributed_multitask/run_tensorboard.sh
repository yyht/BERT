odpscmd=$1
pai_command="
	pai -name tensorboard
		-DsummaryDir='oss://alg-misc/BERT/bert_pretrain/open_domain/pretrain_single_random_green_gan/finetuning/roberta_tiny_green_multilabel_circle_loss_v5_cpc_simclr/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com'
"

echo "${pai_command}"
${odpscmd} -e "${pai_command}"