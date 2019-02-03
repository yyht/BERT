CUDA_VISIBLE_DEVICES="" python restore.py \
	--meta "/data/xuht/politics/model/model.ckpt.meta" \
	--input_checkpoint "/data/xuht/politics/model/model.ckpt" \
	--output_checkpoint "/data/xuht/politics/restore_model/model.ckpt" \
	--output_meta_graph "/data/xuht/politics/restore_model/model.meta"