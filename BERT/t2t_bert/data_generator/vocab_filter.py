import tensorflow as tf
import numpy as np
from collections import Counter

def vocab_filter(corpus, vocab, tokenizer, predefined_vocab_size, corpus_vocab_path):
	dic = Counter()
	for item in corpus:
		token_lst = tokenizer.tokenize(item.text_a)
		for token in token_lst:
			dic[token] += 1

	corpus_vocab = vocab[0:4] # <pad>, <unk>, <s>, </s>
	word_lst = dic.most_common(predefined_vocab_size)

	print(len(word_lst), word_lst[0][0], corpus_vocab)

	corpus_vocab += [item[0] for item in word_lst]
	with tf.gfile.Open(corpus_vocab_path, "w") as fwobj:
		for word in corpus_vocab:
			fwobj.write(word+"\n")