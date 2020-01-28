python ./t2t_bert/offline_debug/itera_data.py \
    --buckets '/data/xuht' \
	--train_file "train.tfrecords"  \
	--batch_size 16 \
	--max_predictions_per_seq 10 \
	--max_length 128