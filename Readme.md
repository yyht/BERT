# 运行event

镜像地址 192.168.3.131 event_predict

docker run -d --rm --name=event_predict -p 9993:9993 -e LANG=C.UTF-8 event_predict supervisord

# training semantic match model see /t2t_tensor/lcqmc

 lcqmc_distributed_order_train.sh for training script
 
 test_lcqmc_distributed_order.py for training .py
 
 app.py for online inference
 
 app_start.sh for sh to start app.py
 
# how to traing:
## make training data 
    import sys,os
    sys.path.append("..")

    import numpy as np
    import tensorflow as tf
    from example import bert_classifier
    from bunch import Bunch
    from example import feature_writer, write_to_tfrecords, classifier_processor
    from data_generator import tokenization
    from data_generator import tf_data_utils
    from model_io import model_io
    from data_generator import pair_data_feature_classifier
    from example.feature_writer import PairClassifierFeatureWriter
    
    tokenizer = tokenization.FullTokenizer(
      vocab_file="/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt", 
    do_lower_case=True)

    import json
    
    classifier_data_api = classifier_processor.LCQMCProcessor()
    classifier_data_api.get_labels("/data/xuht/LCQMC/label_dict.json")
    train_examples = classifier_data_api.get_train_examples("/data/xuht/LCQMC/LCQMC_train.json")
    dev_examples = classifier_data_api.get_train_examples("/data/xuht/LCQMC/LCQMC_dev.json")
    test_examples = classifier_data_api.get_train_examples("/data/xuht/LCQMC/LCQMC_test.json")
    
    write_to_tfrecords.convert_pair_order_classifier_examples_to_features(train_examples,classifier_data_api.label2id,
                                                               128,
                                                               tokenizer,
                                                               "/data/xuht/LCQMC/train.tfrecords"
                                                              )
    write_to_tfrecords.convert_pair_order_classifier_examples_to_features(dev_examples,classifier_data_api.label2id,
                                                               128,
                                                               tokenizer,
                                                               "/data/xuht/LCQMC/dev.tfrecords"
                                                              )
    write_to_tfrecords.convert_pair_order_classifier_examples_to_features(test_examples,classifier_data_api.label2id,
                                                               128,
                                                               tokenizer,
                                                               "/data/xuht/LCQMC/test.tfrecords"
                                                              )
                                                              
## run lcqmc_distributed_order_train.sh 
    with some data and model dir modification such as 
    train_file         "/data/xuht/LCQMC/train.tfrecords"
    dev_file           "/data/xuht/LCQMC/dev.tfrecords"    
    model_output       /data/xuht/LCQMC/model/model_12_5
    max_length         128
    num_classes     2
    epoch,           5
    config_file, init_checkpoint, vocab_file, label_id
    
    
## 中文语义数据：
    192.168.3.134//data/xuht/LCQMC
    192.168.3.134//data/xuht/duplicate_sentence
    192.168.3.131///data/xuht/ccks2018
## 词向量
    192.168.3.134//data/xuht/Chinese_w2v
    192.168.3.134//data/xuht/word2vec_model
## classification
    192.168.3.131//data/xuht/jd_comment    jingdong sentiment
    192.168.3.131//data/xuht/eventy_detection/event/model guangfa event
    192.168.3.131///data/xuht/ChineseSTSListCorpus       Chinese 16w classification data
    
    192.168.3.131//data/xuht/eventy_detection/sentiment   guangfa sentiment
    
## cn ner
    192.168.3.131/data/xuht/ner/test/2018_8_1
    
## mrc
    192.168.3.134//data/xuht/dureader
    
## project data
    192.168.3.131/data/xuht/project/data
    