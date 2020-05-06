import json
import re

fwobj = open('/data/xuht/green_clustering/free_text_pretrain_green.txt')
with open('/data/xuht/chinese_corpus/free_text_pretrain_labert_green_topmine_json_jieba.txt', 'r') as frobj:
	for line in frobj:
		content = json.loads(line.strip())
		text = content['text']
		sentences = re.split(r"([.。!！?？；;，,\s+])", text)
		sentences.append("")
		sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
		for sent in sentences:
			fwobj.write(sent+"\n")
		fwobj.write("\n")
fwobj.close()