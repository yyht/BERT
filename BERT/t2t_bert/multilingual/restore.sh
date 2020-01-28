CUDA_VISIBLE_DEVICES="" python restore.py \
	--meta "/data/xuht/lazada/20190107/model_1_15/model_2.ckpt.meta" \
	--input_checkpoint "/data/xuht/lazada/20190107/model_1_15/model_2.ckpt" \
	--output_checkpoint "/data/xuht/lazada/20190107/restore_model/model_2.ckpt" \
	--output_meta_graph "/data/xuht/lazada/20190107/restore_model/model_2.meta"