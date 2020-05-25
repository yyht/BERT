odpscmd=$1
pai_command="
	pai -name tensorboard
		-DsummaryDir='oss://alg-misc/BERT/sentence_pair/pretrain_gatedcnn_lm/cpc_cls/pretrained_bidgcnn/mnli_xquad_uc_cpc_qh_modified_v2/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com'
"

echo "${pai_command}"
${odpscmd} -e "${pai_command}"