odpscmd=$1
pai_command="
	pai -name tensorboard
		-DsummaryDir='oss://alg-misc/BERT/sentence_pair/pretrain_gatedcnn_lm/cpc_cls/vae_bow/pretrained_light_dgcnn/disc_bi_ligthconv_v3_new/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com'
"

echo "${pai_command}"
${odpscmd} -e "${pai_command}"