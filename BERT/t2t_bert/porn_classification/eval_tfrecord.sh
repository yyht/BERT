CUDA_VISIBLE_DEVICES="0" mpirun -np 1 \
 -H localhost:2 \
python eval_tfrecord.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/porn/model/oqmrc_8.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/porn/label_dict.json" \
 --train_file "/data/xuht/websiteanalyze-data-seqing20180821/global_mining_seqing/mining_info.pkl" \
 --dev_file "/data/xuht/porn/seqing_test_20180821_tf_records" \
 --max_length 128 \
 --model_output "/data/xuht/websiteanalyze-data-seqing20180821/global_mining_seqing/" \
 --epoch 5 \
 --num_classes 5 \
 --batch_size 32 \
 --if_shard "1" \
 --lower_case True \
 --eval_data_file "/data/xuht/websiteanalyze-data-seqing20180821/global_mining_seqing/eval.tfrecords"

