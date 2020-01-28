python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=/data/xuht/bert/exp/XNLI-MT-1.0/XNLI-MT-1.0 \
  --vocab_file=/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/data/xuht/bert/xnli_output1/

