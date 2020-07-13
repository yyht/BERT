odpscmd=$1
model_folder=$2
model_zip=$3
model_type=$4

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*" 

# pai_command="
# # set odps.running.cluster=AY100G;
# # set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
# pai -name tensorflow180_hvd_test 
# 	-project algo_public_dev 
# 	-Dscript='file://${model_zip}'
# 	-DentryFile='./BERT/t2t_bert/distributed_bin/all_reduce_train_eval_api.py' 
# 	-DgpuRequired=400
# 	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_single_sentence_classification/multilingual_lazada'
# 	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
# "
# echo "${pai_command}"
# ${odpscmd} -e "${pai_command}"
# echo "finish..."
# -project algo_public_dev 
# multilingual_abusive
pai_command="
# set odps.running.cluster=AY100G;
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
pai -name tensorflow1120
	-Dscript='file://${model_zip}'
	-DentryFile='./BERT/t2t_bert/distributed_bin/all_reduce_train_eval_api.py' 
	-DgpuRequired=100
	-DjobName='bert_qqp'
	-Dtags='bert'
	-DmaxHungTimeBeforeGCInSeconds=0
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_single_sentence_classification/funnel_transformer_444_rte'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."