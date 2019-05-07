# from tqdm import tqdm
import numpy as np
import re
import tensorflow as tf
from dataset_generator.create_generator import create_generator

def train_eval_input_fn(FLAGS, multi_task_config, mode, epoch):

    def gen():
        g = create_generator(FLAGS, multi_task_config, mode, epoch)
        for example in g:
            yield example

    problem_config = multi_task_config[FLAGS.multi_task_type.split(",")[0]]
    max_seq_len = problem_config["max_length"]

    output_type = {
        'input_ids': tf.int32,
        'input_mask': tf.int32,
        'segment_ids': tf.int32
    }
    output_shapes = {
        'input_ids': [max_seq_len],
        'input_mask': [max_seq_len],
        'segment_ids': [max_seq_len]
    }
   
    if problem_config["lm_augumentation"]:
        output_type.update({
            "masked_lm_positions": tf.int32,
            "masked_lm_ids": tf.int32,
            "masked_lm_weights": tf.float32
        })

        output_shapes.update({
            "masked_lm_positions": [problem_config["max_predictions_per_seq"]],
            "masked_lm_ids": [problem_config["max_predictions_per_seq"]],
            "masked_lm_weights": [problem_config["max_predictions_per_seq"]]
        })
    for problem in FLAGS.multi_task_type.split(","):
        problem_dict = multi_task_config[problem]
        problem_type = multi_task_config[problem]["task_type"]
        
        output_type.update({'%s_loss_multiplier' % problem: tf.int32})
        output_shapes.update({'%s_loss_multiplier' % problem: []})

        output_type.update({'task_id': tf.int32})
        output_shapes.update({'task_id': []})

        if problem_type in ['seq_tag_task']:
            output_type.update({'%s_label_ids' % problem: tf.int32})
            output_shapes.update(
                {'%s_label_ids' % problem: [max_seq_len]})
        elif problem_type in ['cls_task']:
            output_type.update({'%s_label_ids' % problem: tf.int32})
            output_shapes.update({'%s_label_ids' % problem: []})
        elif problem_type in ['seq2seq_tag_task', 'seq2seq_text_task']:
            output_type.update({'%s_label_ids' % problem: tf.int32})
            output_shapes.update(
                {'%s_label_ids' % problem: [max_seq_len]})

            output_type.update({'%s_mask' % problem: tf.int32})
            output_shapes.update(
                {'%s_mask' % problem: [problem_dict.get("decode_max_seq_len", max_seq_len)]})

        elif problem_type in ['pretrain']:
            output_type.update({
                "masked_lm_positions": tf.int32,
                "masked_lm_ids": tf.int32,
                "masked_lm_weights": tf.float32,
                "next_sentence_label_ids": tf.int32
            })

            output_shapes.update({
                "masked_lm_positions": [problem_config["max_predictions_per_seq"]],
                "masked_lm_ids": [problem_config["max_predictions_per_seq"]],
                "masked_lm_weights": [problem_config["max_predictions_per_seq"]],
                "next_sentence_label_ids": []
            })

    tf.logging.info(output_type)
    tf.logging.info(output_shapes)

    print("==begin to build data generator==")

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_type, output_shapes=output_shapes)

    dataset = dataset.batch(FLAGS.batch_size)
    
    return dataset

# def predict_input_fn(input_file_or_list, config: Params, mode='predict'):

#     # if is string, treat it as path to file
#     if isinstance(input_file_or_list, str):
#         inputs = open(input_file_or_list, 'r', encoding='utf8').readlines()
#     else:
#         inputs = input_file_or_list

#     tokenizer = FullTokenizer(config.vocab_file)

#     # data_dict = {}
#     # data_dict['input_ids'] = []
#     # data_dict['input_mask'] = []
#     # data_dict['segment_ids'] = []

#     def gen():
#         data_dict = {}
#         for doc in tqdm(inputs, desc='Processing Inputs'):
#             inputs_a = list(doc)
#             tokens, target = tokenize_text_with_seqs(
#                 tokenizer, inputs_a, None)

#             tokens_a, tokens_b, target = truncate_seq_pair(
#                 tokens, None, target, config.max_seq_len)

#             tokens, segment_ids, target = add_special_tokens_with_seqs(
#                 tokens_a, tokens_b, target)

#             input_mask, tokens, segment_ids, target = create_mask_and_padding(
#                 tokens, segment_ids, target, config.max_seq_len)

#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
#             data_dict['input_ids'] = input_ids
#             data_dict['input_mask'] = input_mask
#             data_dict['segment_ids'] = segment_ids
#             yield data_dict
#     output_type = {
#         'input_ids': tf.int32,
#         'input_mask': tf.int32,
#         'segment_ids': tf.int32
#     }
#     output_shapes = {
#         'input_ids': [config.max_seq_len],
#         'input_mask': [config.max_seq_len],
#         'segment_ids': [config.max_seq_len]
#     }
#     # dataset = tf.data.Dataset.from_tensor_slices(data_dict)
#     dataset = tf.data.Dataset.from_generator(
#         gen, output_types=output_type, output_shapes=output_shapes)
#     dataset = dataset.batch(config.batch_size*2)

#     return dataset