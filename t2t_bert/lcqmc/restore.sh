CUDA_VISIBLE_DEVICES="" python restore.py \
	--meta "/data/xuht/LCQMC/model/model_12_5/oqmrc_4.ckpt.meta" \
	--input_checkpoint "/data/xuht/LCQMC/model/model_12_5/oqmrc_4.ckpt" \
	--output_checkpoint "/data/xuht/porn/restore_model/oqmrc_4.ckpt" \
	--output_meta_graph "/data/xuht/porn/restore_model/oqmrc_4.meta"