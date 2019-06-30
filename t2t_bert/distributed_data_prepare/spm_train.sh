python ./t2t_bert/distributed_data_prepare/spm_train.py \
	--train_file \
	--output_folder \
	--model_prefix \
	--vocab_size 50000 \
	--model_type bpe \
	--character_coverage 0.995
