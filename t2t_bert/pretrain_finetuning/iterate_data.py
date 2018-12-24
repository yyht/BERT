import tensorflow as tf
from bunch import Bunch
import os, sys
sys.path.append("../..")
import sys,os,json
sys.path.append("..")

import numpy as np
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS
from data_generator import tf_data_utils
epoch = 1

flags.DEFINE_string(
    "train_result_file", None,
    "Input TF example files (can be a glob or comma separated).")

def main(_):

    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto()
        import random
        name_to_features = {
                "input_ids":
                    tf.FixedLenFeature([128], tf.int64),
                "input_mask":
                    tf.FixedLenFeature([128], tf.int64),
                "segment_ids":
                    tf.FixedLenFeature([128], tf.int64),
                "masked_lm_positions":
                    tf.FixedLenFeature([5], tf.int64),
                "masked_lm_ids":
                    tf.FixedLenFeature([5], tf.int64),
                "masked_lm_weights":
                    tf.FixedLenFeature([5], tf.float32),
                "label_ids":
                    tf.FixedLenFeature([], tf.int64),
                }

        params = Bunch({})
        params.epoch = epoch
        params.batch_size = 32
        def parse_folder(path):
            files = os.listdir(path)
            output = []
            for file_name in files:
                output.append(os.path.join(path, file_name))
            random.shuffle(output)
            return output
        print(params["batch_size"], "===batch size===")
        jd_test = parse_folder(FLAGS.train_result_file)
        input_fn = tf_data_utils.train_input_fn(jd_test, tf_data_utils._decode_record, name_to_features, params)
        
        sess = tf.Session(config=sess_config)
        
        init_op = tf.group(
                    tf.local_variables_initializer())
        sess.run(init_op)
        
        i = 0
        cnt = 0
        while True:
            try:
                features = sess.run(input_fn)
                i += 1
                cnt += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        print(i*32)

if __name__ == "__main__":
    tf.app.run()