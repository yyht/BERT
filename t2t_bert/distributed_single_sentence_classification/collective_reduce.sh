odpscmd=$1

pai_command="
pai -name tensorflow140_hvd_test 
	-project algo_public_dev 
	-Dscript='file:///Users/xuhaotian/Desktop/my_work/BERT.zip'
	-DentryFile='./BERT/t2t_bert/distributed_bin/collective_reduce_train_eval_api.py' 
	-Dcluster='{\"worker\":{\"count\":2, \"gpu\":100}}'
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_single_sentence_classification/porn_hparameter'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."


