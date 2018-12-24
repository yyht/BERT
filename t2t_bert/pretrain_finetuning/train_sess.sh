CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun -np 4 \
 -H localhost:4 \
python train_sess.py \
 --config_file "/data/xuht/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/jd_comment/label_dict.json" \
 --train_file "/data/xuht/jd_comment/train" \
 --dev_file "/data/xuht/ji_comment/test" \
 --max_length 128 \
 --model_output "/data/xuht/jd_comment/model_12_24" \
 --epoch 8 \
 --num_classes 3 \
 --batch_size 16 \
 --if_shard "1" \
 --max_predictions_per_seq 5 \
 --if_debug 1

