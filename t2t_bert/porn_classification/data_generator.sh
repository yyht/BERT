python data_processor.py \
	--train_file /data/xuht/porn/actual_data/seqing_train_albert_20190228 \
	--test_file /data/xuht/porn/actual_data/seqing_test_albert_20190228 \
	--train_result_file /data/xuht/porn/actual_data/normal/seqing_train_20190228_tf_records \
	--test_result_file  /data/xuht/porn/actual_data/normal/seqing_test_20190228_tf_records\
	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
	--label_id /data/xuht/porn/label_dict.json \
	--lower_case True \
	--max_length 128 \
	--if_rule "no_rule" \
	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \

python data_processor.py \
	--train_file /data/xuht/porn/actual_data/seqing_train_albert_20190228 \
	--test_file /data/xuht/porn/actual_data/seqing_test_albert_20190228 \
	--train_result_file /data/xuht/porn/actual_data/rule/seqing_train_20190228_tf_records \
	--test_result_file  /data/xuht/porn/actual_data/rule/seqing_test_20190228_tf_records\
	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
	--label_id /data/xuht/porn/label_dict.json \
	--lower_case True \
	--max_length 128 \
	--if_rule "rule" \
	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \
