import json
import pinyin
import tensorflow as tf
import re
from gensim.models import word2vec

ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")

def prepare_corpus(input_files):
	sentences = []
	cnt = 0
	for input_file in input_files:
		with tf.gfile.GFile(input_file, "r") as reader:
			while True:
				line = reader.readline()
				
				if not line:
					break
				line = line.strip().split()
				line_pinyin = []
				
				for word in line:
					char_cn = ch_pattern.findall(word)
					if len(char_cn) >= 1:
						line_pinyin.extend(pinyin.get(item, format="strip", delimiter=" "))
					else:
						if len(word) >= 1:
							line_pinyin.extend(word.split())
				if cnt <= 10:
					print(line, line_pinyin)
				cnt += 1

				# Empty lines are used as document delimiters
				if not line:
					# all_documents.append([])
					continue
				sentences.append(line_pinyin)
	return sentences

input_files = '/data/xuht/chinese_corpus/wiki_zh_pretrain.txt,/data/xuht/chinese_corpus/web_text_zh_train.txt,/data/xuht/chinese_corpus/news2016zh_train.txt'
input_files = input_files.split(',')
input_files.append('/data/xuht/chinese_corpus/free_text_pretrain_albert_green_pinyin_modified.txt')
print(input_files)
output_file = '/data/xuht/chinese_corpus/pinyin_embedding.txt'

sentences = prepare_corpus(input_files)
model = word2vec.Word2Vec(sentences, size=128, window=5, min_count=5)  # 训练skip-gram模型,默认window=5
model.wv.save_word2vec_format(output_file, binary=False)