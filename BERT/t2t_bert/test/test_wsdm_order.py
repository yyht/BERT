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

from example import bert_order_classifier


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

flags.DEFINE_string(
    "train_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "dev_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "model_output", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "gpu_id", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "epoch", 5,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "num_classes", 3,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "train_size", 256434,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
    "batch_size", 16,
    "Input TF example files (can be a glob or comma separated).")

def main(_):

    graph = tf.Graph()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    with graph.as_default():
        import json
        
        # config = json.load(open("/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json", "r"))
        
        config = json.load(open(FLAGS.config_file, "r"))

        init_checkpoint = FLAGS.init_checkpoint
        print("===init checkoutpoint==={}".format(init_checkpoint))

        # init_checkpoint = "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        # init_checkpoint = "/data/xuht/concat/model_1/oqmrc.ckpt"
        config = Bunch(config)
        config.use_one_hot_embeddings = True
        config.scope = "bert"
        config.dropout_prob = 0.1
        config.label_type = "single_label"
        # config.loss = "focal_loss"
        
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
        sess = tf.Session()

        num_train_steps = int(
            FLAGS.train_size / FLAGS.batch_size * FLAGS.epoch)
        num_warmup_steps = int(num_train_steps * 0.1)

        num_storage_steps = int(FLAGS.train_size / FLAGS.batch_size)

        print(num_train_steps, num_warmup_steps, "=============")
        
        opt_config = Bunch({"init_lr":1e-5, 
                            "num_train_steps":num_train_steps,
                            "num_warmup_steps":num_warmup_steps})

        model_io_config = Bunch({"fix_lm":False})
        
        model_io_fn = model_io.ModelIO(model_io_config)
        
        num_choice = FLAGS.num_classes
        max_seq_length = FLAGS.max_length

        model_train_fn = bert_order_classifier.classifier_model_fn_builder_v1(
                                config,
                                num_choice,
                                init_checkpoint,
                                model_reuse=None,
                                load_pretrained=True,
                                model_io_fn=model_io_fn,
                                model_io_config=model_io_config,
                                opt_config=opt_config,
                                input_name=["a", "b"])
        
        model_eval_fn = bert_order_classifier.classifier_model_fn_builder_v1(
                                                config, 
                                                num_choice, 
                                                init_checkpoint, 
                                                model_reuse=True, 
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
            return {"accuracy":accuracy, "loss":loss, "pred_label":pred_label, "label_ids":features["label_ids"]}
        
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
        params.epoch = FLAGS.epoch
        params.batch_size = FLAGS.batch_size
        # train_features = tf_data_utils.train_input_fn("/data/xuht/wsdm19/data/train.tfrecords",
        #                             _decode_record, name_to_features, params)
        # eval_features = tf_data_utils.eval_input_fn("/data/xuht/wsdm19/data/dev.tfrecords",
        #                             _decode_record, name_to_features, params)

        train_features = tf_data_utils.train_input_fn(FLAGS.train_file,
                                    _decode_record, name_to_features, params)
        eval_features = tf_data_utils.eval_input_fn(FLAGS.dev_file,
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
            print("test accuracy accuracy {} {} f1 {}".format(total_accuracy/i, 
                accuracy, f1))
            return total_accuracy/ i, f1
        
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
                    if np.mod(i, num_storage_steps) == 0:
                        print(total_loss/cnt)
                        # model_io_fn.save_model(sess, "/data/xuht/wsdm19/data/model_11_15_focal_loss/oqmrc_{}.ckpt".format(int(i/8000)))
                        model_io_fn.save_model(sess, FLAGS.model_output+"/oqmrc_{}.ckpt".format(int(i/num_storage_steps)))
                        
                        print("==successful storing model=={}".format(int(i/num_storage_steps)))
                        total_loss = 0
                        cnt = 0
                except tf.errors.OutOfRangeError:
                    break
        print("===========begin to train============")        
        train_fn(train_op, train_loss)
        print("===========begin to eval============")
        accuracy, f1 = eval_fn(result)
        print("==accuracy {} f1 {}==".format(accuracy, f1))
        # model_io_fn.save_model(sess, "/data/xuht/wsdm19/data/model_11_15_focal_loss/oqmrc.ckpt")

        model_io_fn.save_model(sess, FLAGS.model_output+"/oqmrc.ckpt")

if __name__ == "__main__":
    flags.mark_flag_as_required("eval_data_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("result_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("train_file")
    flags.mark_flag_as_required("dev_file")
    flags.mark_flag_as_required("max_length")
    flags.mark_flag_as_required("model_output")
    flags.mark_flag_as_required("gpu_id")
    flags.mark_flag_as_required("epoch")
    flags.mark_flag_as_required("num_classes")
    tf.app.run()
