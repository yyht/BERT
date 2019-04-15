import tensorflow as tf
from bunch import Bunch
import numpy as np
import os, sys



graph = tf.Graph()
with graph.as_default():
    name_to_features = {
                    "label_id":
                            tf.FixedLenFeature([], tf.int64),
                    "feature":
                            tf.FixedLenFeature([768], tf.float32),
                    "prob":
                            tf.FixedLenFeature([5], tf.float32)
            }

    params = Bunch({})
    params.epoch = 1
    params.batch_size = 32
    
    print(params["batch_size"], "===batch size===")
    # input_fn = tf_data_utils.train_input_fn(jd_test, tf_data_utils._decode_record, name_to_features, params)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example.

        name_to_features = {
                    "input_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "input_mask":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "segment_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "masked_lm_positions":
                            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
                    "masked_lm_ids":
                            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
                    "masked_lm_weights":
                            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
                    "next_sentence_labels":
                            tf.FixedLenFeature([1], tf.int64),
            }

        """
        example = tf.parse_example(record, name_to_features)
        return example

    def train_input_fn(input_file, _parse_fn, name_to_features,
        params, **kargs):

        dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
#         dataset = dataset.shuffle(
#                                 buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
#                                 seed=np.random.randint(0,1e10,1)[0],
#                                 reshuffle_each_iteration=True)
        dataset = dataset.batch(params.get("batch_size", 32))
        dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
        
        dataset = dataset.repeat(params.get("epoch", 100))
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features

    input_fn = train_input_fn(jd_test, _decode_record, name_to_features, params)
    
    sess = tf.Session()
    
    init_op = tf.group(
                tf.local_variables_initializer())
    sess.run(init_op)
    
    i = 0
    cnt = 0
    label_prob, feature = [], []
    while True:
        try:
            features = sess.run(input_fn)
            i += 1
            cnt += 1
            label_prob.extend(features["prob"].tolist())
            feature.extend(features["feature"].tolist())
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
import _pickle as pkl
pkl.dump({"label_prob":label_prob,
         "feature":feature}, 
         open("/data/xuht/porn/clean_data/textcnn/distillation/train_distilaltion.info", "wb"))