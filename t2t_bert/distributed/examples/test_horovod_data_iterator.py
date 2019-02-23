import tensorflow as tf
from bunch import Bunch
import os, sys
sys.path.append("../..")
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
import horovod.tensorflow as hvd
epoch = 1
sess_config = tf.ConfigProto()


graph = tf.Graph()
with graph.as_default():
    name_to_features = {
                    "input_ids":
                            tf.FixedLenFeature([128], tf.int64),
                    "input_mask":
                            tf.FixedLenFeature([128], tf.int64),
                    "segment_ids":
                            tf.FixedLenFeature([128], tf.int64),
                    "label_ids":
                            tf.FixedLenFeature([], tf.int64),
            }

    params = Bunch({})
    params.epoch = epoch
    params.batch_size = 32
    jd_test = "/data/xuht/jd_comment/train.tfrecords"
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
        if_shard = kargs.get("if_shard", "1")

        worker_count = kargs.get("worker_count", 1)
        task_index = kargs.get("task_index", 0)

        dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
        print("==worker_count {}, task_index {}==".format(worker_count, task_index))
        if if_shard == "1":
            dataset = dataset.shard(worker_count, task_index)
        dataset = dataset.shuffle(
                                buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
                                seed=np.random.randint(0,1e10,1)[0],
                                reshuffle_each_iteration=True)
        dataset = dataset.batch(params.get("batch_size", 32))
        dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
        
        
        dataset = dataset.repeat(params.get("epoch", 100))
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features

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
    print(i)