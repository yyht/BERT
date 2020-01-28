python train_bert_lm.py \
 --data_path "/data/xuht/ChineseSTSListCorpus/bert/bert_transformer/" \
 --mask_lm_source_file "/data/xuht/ChineseSTSListCorpus/bert/corpus_bert.txt" \
 --ckpt_dir "/data/xuht/ChineseSTSListCorpus/bert/bert_transformer/checkpoint/" \
 --vocab_size 10000 \
 --gpu 1 \
 --d_model 200 \
 --max_allow_sentence_length 30 \
 --d_k 128 \
 --d_v 128 \
 --word2vec_vocab_path "/data/xuht/Chinese_w2v/Tencent_AILab_ChineseEmbedding/vocab.txt" \
 --word2vec_model_path "/data/xuht/Chinese_w2v/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"


