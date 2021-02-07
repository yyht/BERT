# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")

## Required parameters
flags.DEFINE_string(
    "tfrecord_lst", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("num_gpus", 8, "Total batch size for training.")
flags.DEFINE_integer("num_accumulated_batches", 1, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("weight_decay_rate", 0.9, "The initial learning rate for Adam.")
flags.DEFINE_float("warmup_proportion", 0.1, "The initial learning rate for Adam.")
flags.DEFINE_float("lr_decay_power", 1.0, "The initial learning rate for Adam.")
flags.DEFINE_float("layerwise_lr_decay_power", 0.0, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("train_examples", 2321511.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("log_step_count_steps", 100,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("num_labels", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("keep_checkpoint_max", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("if_multisigmoid", True, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("if_grad_penalty", True, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("do_distributed_training", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,
                if_multisigmoid=False,
                if_grad_penalty=False):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  with tf.variable_scope("cls/gaode/classification"):
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if not if_multisigmoid:
        print("===softmax cross-entropy===")
        probabilities = tf.nn.softmax(logits, axis=-1)
    elif if_multisigmoid:
        print("===multilabel-sigmoid===")
        probabilities = tf.nn.sigmoid(logits)

    return (probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings,
                     if_multisigmoid=False,
                     if_grad_penalty=False,
                     num_towers=1):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, [],
        num_labels, use_one_hot_embeddings,
        if_multisigmoid=if_multisigmoid,
        if_grad_penalty=if_grad_penalty)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    
    output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          export_outputs={
              "output":tf.estimator.export.PredictOutput(
                          {"probabilities": probabilities}
                      )
          })
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  import os
  output_dir = os.path.join(FLAGS.buckets, FLAGS.output_dir)
  init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)

  sess_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=FLAGS.num_labels,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=0,
      num_warmup_steps=0,
      use_one_hot_embeddings=True,
      if_multisigmoid=FLAGS.if_multisigmoid,
      if_grad_penalty=FLAGS.if_grad_penalty,
      num_towers=1)

  receiver_features = {
    "input_ids":tf.placeholder(tf.int32, [None, None], name='input_ids'),
    "input_mask":tf.placeholder(tf.int32, [None, None], name='input_mask'),
    "segment_ids":tf.placeholder(tf.int32, [None, None], name='segment_ids'),
  }

  def serving_input_receiver_fn():
    print(receiver_features, "==input receiver_features==")
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
    return input_fn

  estimator = tf.estimator.Estimator(
              model_fn=model_fn,
              model_dir=output_dir)

  import os
  input_export_dir = os.path.join(output_dir, 'export_dir')

  export_dir = estimator.export_savedmodel(input_export_dir, 
                      serving_input_receiver_fn,
                      checkpoint_path=init_checkpoint)

  print("===Succeeded in exporting saved model==={}".format(export_dir))

if __name__ == "__main__":
  tf.app.run()
