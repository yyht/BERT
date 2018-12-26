CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 \
 -H localhost:2 \
python train_sess.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/jd_comment/label_dict.json" \
 --train_file "/data/xuht/jd_comment/train" \
 --dev_file "/data/xuht/jd_comment/test" \
 --max_length 128 \
 --model_output "/data/xuht/jd_comment/model_12_26" \
 --epoch 5 \
 --num_classes 2 \
 --batch_size 32 \
 --if_shard "1" \
 --max_predictions_per_seq 5 \
 --if_debug 0

