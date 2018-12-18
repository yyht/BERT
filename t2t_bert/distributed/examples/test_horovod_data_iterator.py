import tensorflow as tf
from bunch import Bunch
import os, sys
sys.path.append("../..")
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
import horovod.tensorflow as hvd
epoch = 1
hvd.init()
sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

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
    input_fn = tf_data_utils.train_input_fn(jd_test, tf_data_utils._decode_record, name_to_features, params)
    
    sess = tf.Session(config=sess_config)
    
    init_op = tf.group(
                tf.local_variables_initializer())
    sess.run(init_op)

    sess.run(hvd.broadcast_global_variables(0))
    
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