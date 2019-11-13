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
pai -name tensorflow1120
	-project algo_public_dev 
	-Dscript='file://${model_zip}'
	-DentryFile='./BERT/t2t_bert/offline_debug/run.py'
	-DgpuRequired=400
	-Dtags='bert'
	-DjobName='bert_mrc_pretrain'
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/offline_debug/porn_albert'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/tianyi&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."

