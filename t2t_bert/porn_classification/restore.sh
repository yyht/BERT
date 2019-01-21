CUDA_VISIBLE_DEVICES="0" python restore.py \
	--meta "/data/xuht/porn/model/oqmrc_8.ckpt.meta" \
	--input_checkpoint "/data/xuht/porn/model/oqmrc_8.ckpt" \
	--output_checkpoint "/data/xuht/porn/restore_model/oqmrc_8.ckpt" \