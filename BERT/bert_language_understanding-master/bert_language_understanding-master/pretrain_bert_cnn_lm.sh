python train_bert_lm.py \
 --data_path "/data/xuht/ChineseSTSListCorpus/bert/bert_cnn/" \
 --mask_lm_source_file "/data/xuht/ChineseSTSListCorpus/bert/corpus_bert.txt" \
 --ckpt_dir "/data/xuht/ChineseSTSListCorpus/bert/bert_cnn/checkpoint/" \
 --vocab_size 10000 \
 --gpu 0 \
 --d_model 200 \
 --max_allow_sentence_length 30 \
 --word2vec_vocab_path "/data/xuht/Chinese_w2v/Tencent_AILab_ChineseEmbedding/vocab.txt" \
 --word2vec_model_path "/data/xuht/Chinese_w2v/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"


