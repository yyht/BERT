import sys,os
sys.path.append("..")
from model_io import model_io
import numpy as np
import tensorflow as tf
from example import bert_classifier
from bunch import Bunch
from example import feature_writer, write_to_tfrecords, classifier_processor
from data_generator import tokenization
from data_generator import tf_data_utils

graph = tf.Graph()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
with graph.as_default():
    import json
    
    config = json.load(open("/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json", "r"))
    # init_checkpoint = "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
    init_checkpoint = "/data/xuht/concat/model_1/oqmrc.ckpt"
    config = Bunch(config)
    config.use_one_hot_embeddings = True
    config.scope = "bert"
    config.dropout_prob = 0.1
    config.label_type = "single_label"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    sess = tf.Session()
    
    opt_config = Bunch({"init_lr":1e-5, "num_train_steps":400000})
    model_io_config = Bunch({"fix_lm":False})
    
    model_io_fn = model_io.ModelIO(model_io_config)
    
    num_choice = 3
    max_seq_length = 200

    model_train_fn = bert_classifier.multichoice_model_fn_builder(config, num_choice, init_checkpoint, 
                                            reuse=None, 
                                            load_pretrained=True,
                                            model_io_fn=model_io_fn,
                                            model_io_config=model_io_config, 
                                            opt_config=opt_config)
    
    model_eval_fn = bert_classifier.multichoice_model_fn_builder(config, num_choice, init_checkpoint, 
                                            reuse=True, 
                                            load_pretrained=True,
                                            model_io_fn=model_io_fn,
                                            model_io_config=model_io_config, 
                                            opt_config=opt_config)
    
    def metric_fn(features, logits, loss):
        print(logits.get_shape(), "===logits shape===")
        pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
        prob = tf.nn.softmax(logits)
        accuracy = correct = tf.equal(
            tf.cast(pred_label, tf.int32),
            tf.cast(features["label_ids"], tf.int32)
        )
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return {"accuracy":accuracy, "loss":loss, "pred_label":pred_label, "label_ids":features["label_ids"]}
    
    name_to_features = {
            "input_ids":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "input_mask":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "segment_ids":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "label_ids":
                    tf.FixedLenFeature([], tf.int64),
    }
    
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example.
        """
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        for name in ["input_ids", "input_mask", "segment_ids"]:
            example[name] = tf.reshape(example[name], [-1, max_seq_length])
        return example 

    params = Bunch({})
    params.epoch = 5
    params.batch_size = 6
    train_features = tf_data_utils.train_input_fn("/data/xuht/concat/train.tfrecords",
                                _decode_record, name_to_features, params)
    eval_features = tf_data_utils.eval_input_fn("/data/xuht/concat/test.tfrecords",
                                _decode_record, name_to_features, params)
    
    [train_op, train_loss, train_per_example_loss, train_logits] = model_train_fn(train_features, [], tf.estimator.ModeKeys.TRAIN)
    [_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
    result = metric_fn(eval_features, eval_logits, eval_loss)
    
    model_io_fn.set_saver()
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    def eval_fn(result):
        i = 0
        total_accuracy = 0
        label, label_id = [], []
        while True:
            try:
                eval_result = sess.run(result)
                total_accuracy += eval_result["accuracy"]
                label_id.extend(eval_result["label_ids"])
                label.extend(eval_result["pred_label"])
                i += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        f1 = f1_score(label_id, label, average="macro")
        accuracy = accuracy_score(label_id, label)
        print("test accuracy accuracy {}".format(total_accuracy/i, accuracy))
        return total_accuracy/ i
    
    def train_fn(op, loss):
        i = 0
        cnt = 0
        total_loss = 0.0
        while True:
            try:
                [_, train_loss] = sess.run([op, loss])
                total_loss += train_loss
                i += 1
                cnt += 1
                if np.mod(i, 40000) == 0:
                    print(total_loss/cnt)
                    model_io_fn.save_model(sess, "/data/xuht/concat/model_5_0.1/oqmrc_{}.ckpt".format(int(i/40000)))
                    total_loss = 0
                    cnt = 0
            except tf.errors.OutOfRangeError:
                break
    print("===========begin to train============")        
    train_fn(train_op, train_loss)
    print("===========begin to eval============")
    eval_fn(result)
    model_io_fn.save_model(sess, "/data/xuht/concat/model_5_0.1/oqmrc.ckpt")
