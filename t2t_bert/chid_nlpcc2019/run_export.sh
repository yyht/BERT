odpscmd=$1
model_folder=$2
model_zip=$3

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*" 

pai_command="
pai -name tensorflow180 
	-Dscript='file://${model_zip}'
	-DentryFile='./BERT/t2t_bert/chid_nlpcc2019/export_api.py' 
	-DgpuRequired=100
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/chid_nlpcc2019/chid_export'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."


