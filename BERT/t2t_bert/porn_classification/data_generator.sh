# python data_processor.py \
# 	--train_file /data/xuht/porn/clean_data/train.txt \
# 	--test_file /data/xuht/porn/clean_data/dev.txt \
# 	--train_result_file /data/xuht/porn/clean_data/textcnn/distillation/train_distillation_tfrecords \
# 	--test_result_file  /data/xuht/porn/clean_data/textcnn/distillation/dev_distillation_tfrecords\
# 	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
# 	--label_id /data/xuht/porn/label_dict.json \
# 	--lower_case True \
# 	--max_length 128 \
# 	--if_rule "no_rule" \
# 	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
# 	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
# 	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \

# python data_processor.py \
# 	--train_file /data/xuht/porn/clean_data/dev.txt \
# 	--test_file /data/xuht/porn/clean_data/test.txt \
# 	--train_result_file /data/xuht/porn/clean_data/normal/dev_tfrecords \
# 	--test_result_file  /data/xuht/porn/clean_data/normal/test_tfrecords\
# 	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
# 	--label_id /data/xuht/porn/label_dict.json \
# 	--lower_case True \
# 	--max_length 128 \
# 	--if_rule "no_rule" \
# 	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
# 	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
# 	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \

python data_processor.py \
	--train_file /data/xuht/porn/clean_data/train.txt \
	--test_file /data/xuht/porn/clean_data/dev.txt \
	--train_result_file /data/xuht/porn/clean_data/rule/train_tfrecords \
	--test_result_file  /data/xuht/porn/clean_data/rule/dev_tfrecords\
	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
	--label_id /data/xuht/porn/label_dict.json \
	--lower_case True \
	--max_length 128 \
	--if_rule "rule" \
	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \

python data_processor.py \
	--train_file /data/xuht/porn/clean_data/dev.txt \
	--test_file /data/xuht/porn/clean_data/test.txt \
	--train_result_file /data/xuht/porn/clean_data/rule/dev_tfrecords \
	--test_result_file  /data/xuht/porn/clean_data/rule/test_tfrecords\
	--vocab_file /data/xuht/chinese_L-12_H-768_A-12/vocab.txt \
	--label_id /data/xuht/porn/label_dict.json \
	--lower_case True \
	--max_length 128 \
	--if_rule "rule" \
	--rule_word_dict /data/xuht/porn/rule/rule/phrases.json \
	--rule_word_path /data/xuht/porn/rule/rule/mined_porn_domain_adaptation_v2.txt \
	--rule_label_dict /data/xuht/porn/rule/rule/rule_label_dict.json \
