import sys,os
sys.path.append("..")
import numpy as np
import tensorflow as tf
from example import bert_classifier_estimator
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
from model_io import model_io
from example import feature_writer, write_to_tfrecords, classifier_processor
import json
from data_generator import tokenization
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)

from queue import Queue
class InferAPI(object):
    def __init__(self, config):
        self.config = config
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

    def load_label_dict(self):
        with open(self.config["label2id"], "r") as frobj:
            self.label_dict = json.load(frobj)

    def init_model(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            init_checkpoint = self.config["init_checkpoint"]
            bert_config = json.load(open(self.config["bert_config"], "r"))

            self.model_config = Bunch(bert_config)
            self.model_config.use_one_hot_embeddings = True
            self.model_config.scope = "bert"
            self.model_config.dropout_prob = 0.1
            self.model_config.label_type = "single_label"



            opt_config = Bunch({"init_lr":2e-5, "num_train_steps":1e30, "cycle":False})
            model_io_config = Bunch({"fix_lm":False})

            self.num_classes = len(self.label_dict["id2label"])
            self.max_seq_length = self.config["max_length"]

            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=self.config["bert_vocab"], 
                do_lower_case=True)

            self.sess = tf.Session()
            self.model_io_fn = model_io.ModelIO(model_io_config)
    
            model_fn = bert_classifier_estimator.classifier_model_fn_builder(
                                            self.model_config, 
                                            self.num_classes, 
                                            init_checkpoint, 
                                            reuse=None, 
                                            load_pretrained=True,
                                            model_io_fn=self.model_io_fn,
                                            model_io_config=model_io_config, 
                                            opt_config=opt_config)

            self.estimator = tf.estimator.Estimator(
                        model_fn=model_fn,
                        model_dir=self.config["model_dir"])

    def get_input_features(self, sent_lst):
        input_ids_lst, input_mask_lst, segment_ids_lst = [], [], []
        label_ids_lst = []
        for sent in sent_lst:
            sent = full2half(sent)
            tokens_a = self.tokenizer.tokenize(sent)
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            label_ids = 0

            input_ids_lst.append(input_ids)
            input_mask_lst.append(input_mask)
            segment_ids_lst.append(segment_ids)
            label_ids_lst.append(label_ids)

        return  {"input_ids":np.array(input_ids_lst).astype(np.int32),
            "input_mask":np.array(input_mask_lst).astype(np.int32),
            "segment_ids":np.array(segment_ids_lst).astype(np.int32),
            "label_ids":np.array(label_ids_lst).astype(np.int32)}

    def input_fn(self, input_features):
        dataset = tf.data.Dataset.from_tensor_slices(input_features)
        dataset = dataset.batch(self.config.get("batch_size", 20))
#         iterator = dataset.make_one_shot_iterator()
#         features = iterator.get_next()
        return dataset
    def generate_from_queue(self):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """
        while True:
            yield self.input_queue.get()
            
    def predict_from_queue(self):
        """ Adds a prediction from the model to the output_queue.
        This lives within our 'prediction thread'.
        Note: estimators accept generators as inputs and return generators as output.
        Here, we are iterating through the output generator, which will be 
        populated in lock-step with the input generator.
        """
#      features = self.get_input_features(["据此,订约方同意终止认购协议,而公司及认购方概无责任根据认购协议分別发行及认购股可换股债券。"]*2)
        for i in self.estimator.predict(input_fn=self.queued_predict_input_fn):
#             if self.verbose:
#                 print('Putting in output queue')
            print(i)
            print('Putting in output queue')
            print("===========")
            self.output_queue.put(i)
            
    def queued_predict_input_fn(self):
        """
        Queued version of the `predict_input_fn` in FlowerClassifier.
        Instead of yielding a dataset from data as a parameter, 
        we construct a Dataset from a generator,
        which yields from the input queue.
        """
        
        # Fetch the inputs from the input queue
        output_types = {'input_ids': tf.int32,
                       'input_mask': tf.int32,
                       'segment_ids': tf.int32,
                       'label_ids': tf.int32}
        
        output_shapes = {'input_ids': [None, self.max_seq_length ],
                       'input_mask': [None, self.max_seq_length ],
                       'segment_ids': [None, self.max_seq_length ],
                       'label_ids': [1,]}
        dataset = tf.data.Dataset.from_generator(self.generate_from_queue, output_types=output_types, output_shapes=output_shapes)
        #dataset = dataset.batch(self.config.get("batch_size", 20))
        return dataset
    def predict(self, sent_lst):
        # Get predictions dictionary
        features = dict(self.get_input_features(sent_lst))
        print("call api", self.input_queue.qsize())
        print("call api", self.output_queue.qsize())
        self.input_queue.put(features)
        print("call api", self.input_queue.qsize())
        predictions = self.output_queue.get()  # The latest predictions generator
        print("输出结果后", self.output_queue.qsize())
        return predictions
    def predict_single(self, sent_lst):
        # Get predictions dictionary
        features = dict(self.get_input_features(sent_lst))
#         print("call api", self.input_queue.qsize())
#         print("call api", self.output_queue.qsize())
        self.input_queue.put(features)
#         print("call api", self.input_queue.qsize())
        predictions = self.output_queue.get()  # The latest predictions generator
#        print("输出结果后", self.output_queue.qsize())
        predictions["label"] = self.label_dict["id2label"][str(predictions["pred_label"])]
#         if predictions["label"] == 'other':
#             predictions["label"] = '股票定增'
#             predictions["max_prob"] = 0.0
        return predictions
   
    def predict_batch(self, sen_lst):
        return [self.predict_single([sent]) for sent in sen_lst]
           
    def infer(self, sent_lst):
        with self.graph.as_default():
            for result in self.estimator.predict(input_fn=lambda:  self.input_fn(input_features),
                                                checkpoint_path=self.config["init_checkpoint"]):
                print(result)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
model_config = {
    
    "label2id":"/data/xuht/websiteanalyze-data-seqing20180821/label_dict.json",
    "init_checkpoint":"/data/xuht/websiteanalyze-data-seqing20180821/oqmrc_8.ckpt",
    "bert_config":"/data/xuht/chinese_L-12_H-768_A-12/bert_config.json",
    "max_length":128,
    "bert_vocab":"/data/xuht/chinese_L-12_H-768_A-12/vocab.txt",
    "model_dir":"/data/xuht/websiteanalyze-data-seqing20180821/model"
    
}

api = InferAPI(model_config)
api.load_label_dict()
api.init_model()
from threading import Thread
t = Thread(target=api.predict_from_queue, daemon=True)
t.start()
# while True:
#     import time
#     try:
#         result = api.predict_batch(["据此,订约方同意终止认购协议,而公司及认购方概无责任根据认购协议分別发行及认购股可换股债券。"]*8)
#     except:
#         raise
#     time.sleep(1)


import tornado.ioloop
import tornado.web
import tornado.httpserver
import json


class PredictHandler(tornado.web.RequestHandler):
     def post(self):
        body = json.loads(self.request.body.decode(), encoding="utf-8")
        sentences = body.get("sentences")
        result = api.predict_batch(sentences)
        result = [[[row['label']] for row in result], [[float(row['max_prob'])] for row in result]]
        # print(result)
        return self.write(json.dumps({"code":200, "data":result}, ensure_ascii=False))
def main():
    application = tornado.web.Application([(r"/lxm",PredictHandler),])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.bind(9993)
    http_server.start()
    print("-------------server start-----------------")
    tornado.ioloop.IOLoop.current().start()
if __name__ == "__main__":
    main()


