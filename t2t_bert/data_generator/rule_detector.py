from data_generator import flash_text
import json

class RuleDetector(object):
	def __init__(self, config={}):
		self.config = config
		self.keyword_detector = flash_text.KeywordProcessor()

	def load(self, tokenizer):
		with open(self.config["keyword_path"], "r") as frobj:
			keywords = json.load(frobj)
			for item in keywords:
				self.keyword_detector.add_keyword(tokenizer.tokenize(item["word"]), [item["label"]])
			print("=succeeded in loading keywords==")

		with open(self.config["label_dict"], "r") as frobj:
			self.label_dict = json.laod(frobj)

	def get_rule(self, keywords_found, tokenized_lst):
		rule_type_lst = [self.config["background_label"]] * len(tokenized_lst)
		for item in keywords_found:
			start_pos = item[1]
			end_pos = item[-1]
			for t in range(start_pos, end_pos):
				if isinstance(item[0], list):
					rule_type_lst[t] = item[0][0] # keyword clean_name or keyword_type
				else:
					rule_type_lst[t] = item[0]

		return rule_type_lst

	def get_keyword(self, tokenized_lst):
		keywords_found = self.keyword_detector.extract_keywords(tokenized_lst,
                                                span_info=True)
		return keywords_found

	def infer(self, tokenized_lst):
		keywords_found = self.get_keyword(tokenized_lst)
		rule_type_lst = self.get_rule(keywords_found, tokenized_lst)
		rule_id_lst = []
		for rule_token in rule_type_lst:
			rule_id_lst.append(self.label_dict["label2id"][rule_token])
		return rule_id_lst