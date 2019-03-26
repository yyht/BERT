odpscmd=$1
model_folder=$2
model_zip=$3
model_type=$4
file_name=requirements.txt

# if [ ! -f ${model_zip} ]
# then
#   rm ${model_zip}
# fi

# zip -r ${model_zip} ${model_folder}"/"${file_name} ${model_folder} -x "*.DS_Store,*.git*" 

pai_command="
pai -name tensorflow112_boosting
	-project algo_public_dev 
	-Dscript='file://${model_zip}'
	-DentryFile='./BERT/t2t_bert/distributed_bin/soar_train_eval_api.py' 
	-Dcluster='{\"worker\":{\"count\":1,\"gpu\":400,\"memory\":50000}}'
	-DrdmaRequired='no'
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_pair_sentence_classification/soar_bert_lcqmc'
	-Dbuckets='oss://alg-misc/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."