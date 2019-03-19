python ./t2t_bert/distributed_pair_sentence_classification/restore.py \
	--buckets "/data/xuht" \
	--meta "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/model.ckpt-186501.meta" \
	--input_checkpoint "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/model.ckpt-186501" \
	--output_checkpoint "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/restore/model.ckpt-186501" \
	--output_meta_graph "lcqmc/data/model/estimator/distillation/match_pyramid_0317_focal_loss_distillation_0.9_mask/restore/model.ckpt-186501.meta"