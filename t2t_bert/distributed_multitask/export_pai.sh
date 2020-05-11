odpscmd=$1
model_folder=$2
model_zip=$3

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*" 

pai_command="
pai -name tensorflow1120 
	-Dscript='file://${model_zip}'
	-DentryFile='./BERT/t2t_bert/distributed_multitask/export_api.py' 
	-DgpuRequired=0
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_multitask/export_multilingual_light_embedding_new'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."


