"""A simple example to test the a DistributionStrategy with Estimators.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

import sys,os
import numpy as np
import tensorflow as tf
from bunch import Bunch
import json

father_path = os.path.join(os.getcwd())
print(father_path, "==father path==")

def find_bert(father_path):
  if father_path.split("/")[-1] == "BERT":
    return father_path

  output_path = ""
  for fi in os.listdir(father_path):
    if fi == "BERT":
      output_path = os.path.join(father_path, fi)
      break
    else:
      if os.path.isdir(os.path.join(father_path, fi)):
        find_bert(os.path.join(father_path, fi))
      else:
        continue
  return output_path

bert_path = find_bert(father_path)
t2t_bert_path = os.path.join(bert_path, "t2t_bert")
sys.path.extend([bert_path, t2t_bert_path])

print(sys.path)


flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.DEBUG)

flags.DEFINE_string("buckets", "", "oss buckets")

flags.DEFINE_string(
  "config_file", None,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "init_checkpoint", None,
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

flags.DEFINE_integer(
  "epoch", 5,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "num_classes", 5,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "train_size", 1402171,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "batch_size", 32,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "model_type", None,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "if_shard", None,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "eval_size", 1000,
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "opt_type", "ps_sync",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "is_debug", "0",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
  "run_type", "0",
  "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
  "num_gpus", 2, 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "distribution_strategy", "MirroredStrategy", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "cross_tower_ops_type", "paisoar",
  "the CollectiveAllReduceStrategy cross_tower_ops_type"
  )

flags.DEFINE_string(
  "parse_type", "parse_single", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "rule_model", "normal", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "profiler", "normal", 
  "the required num_gpus"
  )


flags.DEFINE_string(
  "train_op", "adam_weight_decay_exclude", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "running_type", "train", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "load_pretrained", "no", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "w2v_path", "",
  "pretrained w2v"
  )

flags.DEFINE_string(
  "with_char", "no_char",
  "pretrained w2v"
  )

flags.DEFINE_string(
  "input_target", "", 
  "the required num_gpus"
  )

flags.DEFINE_string(
  "decay", "no",
  "pretrained w2v"
  )

flags.DEFINE_string(
  "warmup", "no",
  "pretrained w2v"
  )

flags.DEFINE_string(
  "distillation", "normal",
  "if apply distillation"
  )

flags.DEFINE_float(
  "temperature", 2.0,
  "if apply distillation"
  )

flags.DEFINE_float(
  "distillation_ratio", 1.0,
  "if apply distillation"
  )

flags.DEFINE_integer(
  "num_hidden_layers", 12,
  "if apply distillation"
  )

flags.DEFINE_string(
  "task_type", "single_sentence_classification",
  "if apply distillation"
  )

flags.DEFINE_string(
  "classifier", "order_classifier",
  "if apply distillation"
  )

flags.DEFINE_string(
  "output_layer", "interaction",
  "if apply distillation"
  )

flags.DEFINE_integer(
  "char_limit", 5,
  "if apply distillation"
  )

flags.DEFINE_string(
  "mode", "single_task",
  "if apply distillation"
  )

flags.DEFINE_string(
  "multi_task_type", "wsdm",
  "if apply distillation"
  )

flags.DEFINE_string(
  "multi_task_config", "wsdm",
  "if apply distillation"
  )

flags.DEFINE_string(
  "task_invariant", "no",
  "if apply distillation"
  )

flags.DEFINE_float(
  "init_lr", 5e-5,
  "if apply distillation"
  )

flags.DEFINE_string(
  "multitask_balance_type", "data_balanced",
  "if apply distillation"
  )

flags.DEFINE_integer(
  "prefetch", 0,
  "if apply distillation"
  )

# if FLAGS.loglevel == "info":
# tf.logging.set_verbosity(tf.logging.INFO)
# elif FLAGS.loglevel == "debug":
#   tf.logging.set_verbosity(tf.logging.DEBUG)
# else:
#   tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib

from dataset_generator.input_fn import train_eval_input_fn 

def build_model_fn_optimizer():
  """Simple model_fn with optimizer."""
  # TODO(anjalisridhar): Move this inside the model_fn once OptimizerV2 is
  # done?
  optimizer = tf.train.GradientDescentOptimizer(0.2)

  def model_fn(features, labels, mode):  # pylint: disable=unused-argument
    """model_fn which uses a single unit Dense layer."""
    # You can also use the Flatten layer if you want to test a model without any
    # weights.
    layer = tf.layers.Dense(1, use_bias=True)
    logits = tf.reduce_mean(layer(tf.cast(features["input_ids"], tf.float32)))/1000

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {"logits": logits}
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    def loss_fn():
      y = tf.reshape(logits, []) - tf.constant(1.)
      return y * y

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss_fn())

    assert mode == tf.estimator.ModeKeys.TRAIN

    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss_fn(), global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss_fn(), train_op=train_op)

  return model_fn

def data_generator():
  print("==data path==", os.path.join(FLAGS.buckets, "test","generator_test"))
  with tf.gfile.Open(os.path.join(FLAGS.buckets, "test","generator_test"), "r") as frobj:
    for i in frobj:
      print(type(i.strip()), i.strip(), "===========")
      try:
        s = float(i.strip())
      except:
        s = 1.0
      yield [[float(s)]]


def main():
  cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
  distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=2, cross_tower_ops=cross_tower_ops)
  config = tf.estimator.RunConfig(train_distribute=distribution)
  multi_task_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))
  def input_fn():
    features = tf.data.Dataset.from_tensors([[1.]]).repeat(10)
    labels = tf.data.Dataset.from_tensors([1.]).repeat(10)
    return tf.data.Dataset.zip((features, labels))

  def input_fn_generator():
    features = tf.data.Dataset.from_generator(data_generator, tf.float32, tf.TensorShape([None,1]))
    labels = tf.data.Dataset.from_generator(data_generator, tf.float32, tf.TensorShape([None,1]))
    dataset = tf.data.Dataset.zip((features, labels)).repeat(10)
    return dataset
  def input_fn_generator():
    dataset = train_eval_input_fn(FLAGS, multi_task_config, "train", 0)
    return dataset

  model_fn = build_model_fn_optimizer()

  est = tf.estimator.Estimator(model_fn=model_fn, config=config)
  print("==begin to train==")
  est.train(input_fn=input_fn_generator, max_steps=1000)

main()