python ./t2t_bert/pretrain_finetuning/iterate_data.py \
    --buckets '/data/xuht' \
	--train_file "chunk_0.tfrecords"  \
	--batch_size 1 \
	--max_predictions_per_seq 5 \
	--max_length 384