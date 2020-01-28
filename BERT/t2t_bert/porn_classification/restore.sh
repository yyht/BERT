CUDA_VISIBLE_DEVICES="" python restore.py \
	--meta "/data/xuht/porn/model/oqmrc_8.ckpt.meta" \
	--input_checkpoint "/data/xuht/porn/model/oqmrc_8.ckpt" \
	--output_checkpoint "/data/xuht/porn/restore_model/oqmrc_8.ckpt" \
	--output_meta_graph "/data/xuht/porn/restore_model/oqmrc_8.meta"