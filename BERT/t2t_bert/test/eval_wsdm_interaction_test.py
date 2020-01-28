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
import json

from example import feature_writer, write_to_tfrecords, classifier_processor
from data_generator import tokenization
from data_generator import tf_data_utils

from example import bert_interaction_classifier

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"eval_data_file", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"output_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "label_id", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "max_length", 128,
    "Input TF example files (can be a glob or comma separated).")

graph = tf.Graph()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with graph.as_default():
    import json
    
    # config = json.load(open("/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json", "r"))
    config = json.load(open(FLAGS.config_file, "r"))
    # init_checkpoint = "/data/xuht/concat/model_5/oqmrc_{}.ckpt".format(5)
    # init_checkpoint = "/data/xuht/concat/model_5_0.1_inf/oqmrc_{}.ckpt".format(2)
    init_checkpoint = FLAGS.init_checkpoint
    print("===init checkoutpoint==={}".format(init_checkpoint))
    # init_checkpoint = "/data/xuht/concat/model_1/oqmrc.ckpt"
    # init_checkpoint = "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
#     init_checkpoint = "/data/xuht/ai_challenge_cqmrc/bert/concat/model/oqmrc.ckpt"
    config = Bunch(config)
    config.use_one_hot_embeddings = True
    config.scope = "bert"
    config.dropout_prob = 0.2
    config.label_type = "single_label"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    sess = tf.Session()
    
    opt_config = Bunch({"init_lr":1e-5, "num_train_steps":80000})
    model_io_config = Bunch({"fix_lm":False})
    
    model_io_fn = model_io.ModelIO(model_io_config)
    
    num_choice = 3
    max_seq_length = FLAGS.max_length
    
    model_eval_fn = bert_interaction_classifier.classifier_model_fn_builder(
                                            config, 
                                            num_choice, 
                                            init_checkpoint, 
                                            model_reuse=None, 
                                            load_pretrained=True,
                                            model_io_fn=model_io_fn,
                                            model_io_config=model_io_config, 
                                            opt_config=opt_config,
                                            input_name=["a", "b"])
    
    def metric_fn(features, logits, loss):
        print(logits.get_shape(), "===logits shape===")
        pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
        prob = tf.nn.softmax(logits)
        accuracy = correct = tf.equal(
            tf.cast(pred_label, tf.int32),
            tf.cast(features["label_ids"], tf.int32)
        )
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return {"accuracy":accuracy, "loss":loss, 
                "pred_label":pred_label, "label_ids":features["label_ids"],
                "pred_prob":prob}
    
    name_to_features = {
            "input_ids_a":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask_a":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids_a":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_ids_b":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask_b":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids_b":
                    tf.FixedLenFeature([max_seq_length], tf.int64),
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
        return example 

    params = Bunch({})
    params.epoch = 2
    params.batch_size = 32

    # eval_features = tf_data_utils.eval_input_fn("/data/xuht/concat/test.tfrecords",
    #                             _decode_record, name_to_features, params)

    eval_features = tf_data_utils.eval_input_fn(FLAGS.output_file,
                                _decode_record, name_to_features, params)
    
    [_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
    result = metric_fn(eval_features, eval_logits, eval_loss)
    
    model_io_fn.set_saver()
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    def eval_fn(result):
        i = 0
        total_accuracy = 0
        total_loss = 0.0
        pred_prob = []
        label, label_id = [], []
        while True:
            try:
                eval_result = sess.run(result)
                total_accuracy += eval_result["accuracy"]
                label_id.extend(eval_result["label_ids"])
                label.extend(eval_result["pred_label"])
                total_loss += eval_result["loss"]
                pred_prob.extend(eval_result["pred_prob"])
                i += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        f1 = f1_score(label_id, label, average="macro")
        accuracy = accuracy_score(label_id, label)
        print("test accuracy {} accuracy {} loss {} f1 {}".format(total_accuracy/i, 
            accuracy, total_loss/i, f1))
        return pred_prob
    
    print("===========begin to eval============")
    pred_prob = eval_fn(result)
    with open("/data/xuht/tmp.txt", "w") as fwobj:
        list_output = []
        for item in pred_prob:
            list_output.append(item.tolist())
        json.dump(list_output, fwobj)
