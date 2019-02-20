odpscmd=$1

pai_command="
set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
pai -name tensorflow140_hvd_test 
	-project algo_public_dev 
	-Dscript='file:///Users/xuhaotian/Desktop/my_work/BERT.zip'
	-DentryFile='./BERT/t2t_bert/distributed_bin/all_reduce_train_eval_api.py' 
	-DgpuRequired=800 
	-DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/BERT/t2t_bert/distributed_single_sentence_classification/jd_hvd_hparameter'
	-Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."


