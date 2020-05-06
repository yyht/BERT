
import jieba
import re

# a = "你在北京 我,你是谁？在干嘛."
# sentences = re.split(r"([.。!！?？；;])", a)
# sentences.append("")
# sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]


fwobj = open('/data/xuht/chinese_corpus/free_text_pretrain_albert_green_pinyin_modified.txt', 'w')
with open('/data/xuht/chinese_corpus/free_text_pretrain_albert_green_pinyin.txt', 'r') as frobj:
	for i, line in enumerate(frobj):
		if i == 0:
			continue
		else:
			sentences = re.split(r"([.。!！?？；;])", line.strip())
			sentences.append("")
			sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
			for sent in sentences:
				fwobj.write(" ".join(jieba.cut(sent))+"\n")
fwobj.close()