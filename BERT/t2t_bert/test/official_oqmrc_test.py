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

graph = tf.Graph()
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
with graph.as_default():
    import json

    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, 
        do_lower_case=True)

    classifier_data_api = classifier_processor.MultiChoiceProcessor()

    eval_examples = classifier_data_api.get_eval_examples(FLAGS.eval_data_file)

    print(eval_examples[0].qas_id)

    write_to_tfrecords.convert_multichoice_examples_to_features(eval_examples,{},
                                                           200,
                                                           tokenizer,
                                                           FLAGS.output_file)

    config = json.load(open(FLAGS.config_file, "r"))
    init_checkpoint = FLAGS.init_checkpoint

    config = Bunch(config)
    config.use_one_hot_embeddings = True
    config.scope = "bert"
    config.dropout_prob = 0.2
    config.label_type = "single_label"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sess = tf.Session()
    
    opt_config = Bunch({"init_lr":1e-5, "num_train_steps":80000})
    model_io_config = Bunch({"fix_lm":False})
    
    model_io_fn = model_io.ModelIO(model_io_config)
    
    num_choice = 3
    max_seq_length = 200

    model_eval_fn = bert_classifier.multichoice_model_fn_builder(config, num_choice, init_checkpoint, 
                                            reuse=None, 
                                            load_pretrained=True,
                                            model_io_fn=model_io_fn,
                                            model_io_config=model_io_config, 
                                            opt_config=opt_config)
    
    def metric_fn(features, logits):
        print(logits.get_shape(), "===logits shape===")
        pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return {"pred_label":pred_label, "qas_id":features["qas_id"]}
    
    name_to_features = {
            "input_ids":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "input_mask":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "segment_ids":
                    tf.FixedLenFeature([max_seq_length*num_choice], tf.int64),
            "label_ids": 
                    tf.FixedLenFeature([], tf.int64),
            "qas_id":
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
    params.epoch = 2
    params.batch_size = 6

    eval_features = tf_data_utils.eval_input_fn(FLAGS.output_file,
                                _decode_record, name_to_features, params)
    
    [_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
    result = metric_fn(eval_features, eval_logits)
    
    model_io_fn.set_saver()
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    def eval_fn(result):
        i = 0
        pred_label, qas_id = [], []
        while True:
            try:
                eval_result = sess.run(result)
                pred_label.extend(eval_result["pred_label"])
                qas_id.extend(eval_result["qas_id"])
                i += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        return pred_label, qas_id
    
    print("===========begin to eval============")
    [pred_label, qas_id] = eval_fn(result)
    result = dict(zip(qas_id, pred_label))

    print(len(result), "=====valid result======")

    with tf.gfile.Open(FLAGS.eval_data_file, "r") as frobj:
        qas_answer = {}
        for line in frobj:
            content = json.loads(line.strip())
            qas_answer[int(content["query_id"])] = tokenization.convert_to_unicode(content["alternatives"]).split("|")

    with tf.gfile.Open(FLAGS.result_file, "w") as fwobj:
        cnt = 0
        for index, key in enumerate(qas_answer):
            if key in result:
                cnt += 1
                if index == 10:
                    print("==index=={}".format(index))
                pred_ans = qas_answer[key][result[key]]
                fwobj.write("\t".join([str(key), pred_ans])+"\n")
            else:
                pred_ans = qas_answer[key][0]
                fwobj.write("\t".join([str(key), pred_ans])+"\n")

    print(len(result), cnt, len(qas_answer), "======num of results from model==========")

if __name__ == "__main__":
    flags.mark_flag_as_required("eval_data_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("result_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()


