odpscmd=$1
model_folder=$2
model_zip=$3
model_type=$4

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*" 

pai_command="
set odps.running.cluster=AY100G;
pai -name tensorflow1120
	-project algo_public_dev
	-Dtags='bert'
	-DjobName='bert_finetuning'
	-DmaxHungTimeBeforeGCInSeconds=0
	-Dscript='file:///Users/xuhaotian/Desktop/my_work/BERT.zip'
	-DentryFile='./BERT/t2t_bert/distributed_bin/collective_reduce_train_eval_api.py'
	-Dcluster='{\"worker\":{\"count\":2, \"gpu\":200}}'
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_single_sentence_classification/bert_tiny_youku_title_comment'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."

