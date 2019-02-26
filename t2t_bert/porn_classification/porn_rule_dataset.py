import sys,os
sys.path.append("..")

import numpy as np
import tensorflow as tf
from example import bert_classifier
from bunch import Bunch
from example import feature_writer, write_to_tfrecords, classifier_processor
from porn_classification import classifier_processor
from data_generator import tokenization
from data_generator import tf_data_utils
from model_io import model_io
import json

tokenizer = tokenization.FullTokenizer(
      vocab_file="/data/xuht/chinese_L-12_H-768_A-12/vocab.txt", 
    do_lower_case=True)

with open("/data/xuht/websiteanalyze-data-seqing20180821/data/rule/mined_porn_domain_adaptation_v2.txt", "r") as frobj:
    lines = frobj.read().splitlines()
    freq_dict = []
    for line in lines:
        content = line.split("&&&&")
        word = "".join(content[0].split("&"))
        label = "rule"
        tmp = {}
        tmp["word"] = word
        tmp["label"] = "rule"
        freq_dict.append(tmp)
    print(len(freq_dict))
    json.dump(freq_dict, open("/data/xuht/websiteanalyze-data-seqing20180821/data/rule/phrases.json", "w"))
from data_generator import rule_detector

label_dict = {"label2id":{"正常":0,"rule":1}, "id2label":{0:"正常", 1:"rule"}}
json.dump(label_dict, open("/data/xuht/websiteanalyze-data-seqing20180821/data/rule/rule_label_dict.json", "w"))

rule_config = {
    "keyword_path":"/data/xuht/websiteanalyze-data-seqing20180821/data/rule/phrases.json",
    "background_label":"正常",
    "label_dict":"/data/xuht/websiteanalyze-data-seqing20180821/data/rule/rule_label_dict.json"
}
rule_api = rule_detector.RuleDetector(rule_config)
rule_api.load(tokenizer)

classifier_data_api = classifier_processor.PornClassifierProcessor()
classifier_data_api.get_labels("/data/xuht/websiteanalyze-data-seqing20180821/data/label_dict.json")

train_examples = classifier_data_api.get_train_examples(
    "/data/xuht/websiteanalyze-data-seqing20180821/data/seqing_train_20180821")

write_to_tfrecords.convert_classifier_examples_with_rule_to_features(train_examples,
                                                        classifier_data_api.label2id,
                                                        128,
                                                        tokenizer,
                                                        rule_api,
                                                        "/data/xuht/websiteanalyze-data-seqing20180821/data/rule/train.tfrecords")

test_examples = classifier_data_api.get_train_examples("data/xuht/websiteanalyze-data-seqing20180821/data/seqing_eval_20180821")
write_to_tfrecords.convert_classifier_examples_with_rule_to_features(test_examples,
                                                        classifier_data_api.label2id,
                                                        128,
                                                        tokenizer,
                                                        rule_api,
                                                        "/data/xuht/websiteanalyze-data-seqing20180821/data/rule/test.tfrecords")