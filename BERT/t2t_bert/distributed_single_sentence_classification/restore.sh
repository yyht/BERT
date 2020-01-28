CUDA_VISIBLE_DEVICES="" python ./t2t_bert/distributed_pair_sentence_classification/restore.py \
	--buckets "/data/xuht" \
	--meta "porn/clean_data/textlstm/model/estimator/distillation/all_reduce_4_adam_weight_0314_temperature_2/model.ckpt-639755.meta" \
	--input_checkpoint "porn/clean_data/textlstm/model/estimator/distillation/all_reduce_4_adam_weight_0314_temperature_2/model.ckpt-639755" \
	--output_checkpoint "porn/clean_data/textlstm/model/estimator/distillation/all_reduce_4_adam_weight_0314_temperature_2/restore/model.ckpt-639755" \
	--output_meta_graph "porn/clean_data/textlstm/model/estimator/distillation/all_reduce_4_adam_weight_0314_temperature_2/restore/model.ckpt-639755.meta"