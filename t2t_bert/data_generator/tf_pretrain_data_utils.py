"""Create input function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import collections

import numpy as np
import tensorflow as tf
from data_generator.tf_data_utils_confusion_set import confusion_set_sample

def check_tf_version():
    version = tf.__version__
    print("==tf version==", version)
    if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
        return True
    else:
        return False

# flags.DEFINE_integer("vocab_size", default=None, help="")
# flags.DEFINE_integer("unk_id", default=None, help="")
# flags.DEFINE_integer("bos_id", default=None, help="")
# flags.DEFINE_integer("eos_id", default=None, help="")
# flags.DEFINE_integer("cls_id", default=None, help="")
# flags.DEFINE_integer("sep_id", default=None, help="")
# flags.DEFINE_integer("pad_id", default=None, help="")
# flags.DEFINE_integer("mask_id", default=None, help="")
# flags.DEFINE_integer("eod_id", default=None, help="")
# flags.DEFINE_integer("eop_id", default=None, help="")

# flags.DEFINE_integer("seg_id_a", default=0, help="segment id of segment A.")
# flags.DEFINE_integer("seg_id_b", default=1, help="segment id of segment B.")
# flags.DEFINE_integer("seg_id_cls", default=2, help="segment id of cls.")
# flags.DEFINE_integer("seg_id_pad", default=0, help="segment id of pad.")

special_symbols_mapping = collections.OrderedDict([
        ("<unk>", "unk_id"),
        ("<s>", "bos_id"),
        ("</s>", "eos_id"),
        ("<cls>", "cls_id"),
        ("<sep>", "sep_id"),
        ("<pad>", "pad_id"),
        ("<mask>", "mask_id"),
        ("<eod>", "eod_id"),
        ("<eop>", "eop_id")
])

def prepare_text_infilling(input_ids, duplicate_ids=103):
    input_left_shift = tf.concat((input_ids[1:], [0]), axis=0)
    mask_left_shift = tf.logical_or(tf.not_equal(input_ids - input_left_shift, 0), tf.not_equal(input_ids, duplicate_ids))
    dup_mask = tf.concat(([True], mask_left_shift[:-1]), axis=0)
    dup_input_ids_out = tf.boolean_mask(input_ids, dup_mask)
    return dup_input_ids_out, dup_mask
    
def _get_boundary_indices(tokenizer, seg, reverse=False):
    """Get all boundary indices of whole words."""
    seg_len = len(seg)
    if reverse:
        seg = np.flip(seg, 0)

    boundary_indices = []
    for idx, token in enumerate(seg):
        if tokenizer.is_start_token(token) and not tokenizer.is_func_token(token):
            boundary_indices.append(idx)
    boundary_indices.append(seg_len)

    if reverse:
        boundary_indices = [seg_len - idx for idx in boundary_indices]

    return boundary_indices


def setup_special_ids(FLAGS, tokenizer):
    """Set up the id of special tokens."""
    FLAGS.vocab_size = tokenizer.get_vocab_size()
    tf.logging.info("Set vocab_size: %d.", FLAGS.vocab_size)
    for sym, sym_id_str in special_symbols_mapping.items():
        try:
            sym_id = tokenizer.get_token_id(sym)
            setattr(FLAGS, sym_id_str, sym_id)
            tf.logging.info("Set %s to %d.", sym_id_str, sym_id)
        except KeyError:
            tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)


def format_filename(prefix, suffix, seq_len, uncased):
    """Format the name of the tfrecord/meta file."""
    seq_str = "seq-{}".format(seq_len)
    if uncased:
        case_str = "uncased"
    else:
        case_str = "cased"

    file_name = "{}.{}.{}.{}".format(prefix, seq_str, case_str, suffix)

    return file_name


def convert_example(example, use_bfloat16=False):
    """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
    for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
            val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
            val = tf.cast(val, tf.int32)
        if use_bfloat16 and val.dtype == tf.float32:
            val = tf.cast(val, tf.bfloat16)

        example[key] = val


def sparse_to_dense(example):
    """Convert sparse feature to dense ones."""
    for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
            val = tf.sparse.to_dense(val)
        example[key] = val

    return example


# def read_docs(FLAGS, file_path, tokenizer):
#   """Read docs from a file separated by empty lines."""
#   # working structure used to store each document
#   all_docs = []
#   doc, end_of_doc = [], False

#   line_cnt = 0
#   tf.logging.info("Start processing %s", file_path)
#   for line in tf.io.gfile.GFile(file_path):
#       if line_cnt % 100000 == 0:
#           tf.logging.info("Loading line %d", line_cnt)

#       if not line.strip():
#           # encounter an empty line (end of a document)
#           end_of_doc = True
#           cur_sent = []
#       else:
#           cur_sent = tokenizer.convert_text_to_ids(line.strip())

#       if cur_sent:
#           line_cnt += 1
#           doc.append(np.array(cur_sent))

#       # form a doc
#       if end_of_doc or sum(map(len, doc)) >= FLAGS.max_doc_len:
#           # only retain docs longer than `min_doc_len`
#           doc_len = sum(map(len, doc))
#           if doc_len >= max(FLAGS.min_doc_len, 1):
#               all_docs.append(doc)

#           # refresh working structs
#           doc, end_of_doc = [], False

#   # deal with the leafover if any
#   if doc:
#       # only retain docs longer than `min_doc_len`
#       doc_len = sum(map(len, doc))
#       if doc_len >= max(FLAGS.min_doc_len, 1):
#           all_docs.append(doc)

#   tf.logging.info("Finish %s with %d docs from %d lines.", file_path,
#                                   len(all_docs), line_cnt)

#   return all_docs

# flags.DEFINE_enum("sample_strategy", default="single_token",
#                                   enum_values=["single_token", "whole_word", "token_span",
#                                                            "word_span"],
#                                   help="Stragey used to sample prediction targets.")
# flags.DEFINE_bool("shard_across_host", default=True,
#                                   help="Shard files across available hosts.")

# flags.DEFINE_float("leak_ratio", default=0.1,
#                                    help="Percent of masked positions that are filled with "
#                                    "original tokens.")
# flags.DEFINE_float("rand_ratio", default=0.1,
#                                    help="Percent of masked positions that are filled with "
#                                    "random tokens.")

# flags.DEFINE_integer("max_tok", default=5,
#                                        help="Maximum number of tokens to sample in a span."
#                                        "Effective when token_span strategy is used.")
# flags.DEFINE_integer("min_tok", default=1,
#                                        help="Minimum number of tokens to sample in a span."
#                                        "Effective when token_span strategy is used.")

# flags.DEFINE_integer("max_word", default=5,
#                                        help="Maximum number of whole words to sample in a span."
#                                        "Effective when word_span strategy is used.")
# flags.DEFINE_integer("min_word", default=1,
#                                        help="Minimum number of whole words to sample in a span."
#                                        "Effective when word_span strategy is used.")


# def parse_files_to_dataset(parser, file_paths, split, num_hosts,
#                                                    host_id, num_core_per_host, bsz_per_core,
#                                                    num_threads=256, shuffle_buffer=20480):
#   """Parse a list of file names into a single tf.dataset."""

#   def get_options():
#       options = tf.data.Options()
#       # Forces map and interleave to be sloppy for enhance performance.
#       options.experimental_deterministic = False
#       return options

#   if FLAGS.shard_across_host and len(file_paths) >= num_hosts:
#       tf.logging.info("Shard %d files across %s hosts.", len(file_paths),
#                                       num_hosts)
#       file_paths = file_paths[host_id::num_hosts]
#   tf.logging.info("Host %d/%d handles %d files", host_id, num_hosts,
#                                   len(file_paths))

#   assert split == "train"
#   dataset = tf.data.Dataset.from_tensor_slices(file_paths)

#   # file-level shuffle
#   if len(file_paths) > 1:
#       tf.logging.info("Perform file-level shuffle with size %d", len(file_paths))
#       dataset = dataset.shuffle(len(file_paths))

#   # `cycle_length` is the number of parallel files that get read.
#   cycle_length = min(num_threads, len(file_paths))
#   tf.logging.info("Interleave %d files", cycle_length)

#   dataset = dataset.interleave(
#           tf.data.TFRecordDataset,
#           num_parallel_calls=cycle_length,
#           cycle_length=cycle_length)
#   dataset.with_options(get_options())

#   tf.logging.info("Perform sample-level shuffle with size %d", shuffle_buffer)
#   dataset = dataset.shuffle(buffer_size=shuffle_buffer)

#   # Note: since we are doing online preprocessing, the parsed result of
#   # the same input at each time will be different. Thus, cache processed data
#   # is not helpful. It will use a lot of memory and lead to contrainer OOM.
#   # So, change to cache non-parsed raw data instead.
#   dataset = dataset.map(
#           parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
#   dataset = dataset.batch(bsz_per_core, drop_remainder=True)
#   dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

#   return dataset


def _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len, num_predict):
    """Turn beg and end indices into actual mask."""
    non_func_mask = tf.logical_and(
            tf.not_equal(inputs, FLAGS.sep_id),
            tf.not_equal(inputs, FLAGS.cls_id))
    all_indices = tf.where(
            non_func_mask,
            tf.range(tgt_len, dtype=tf.int64),
            tf.constant(-1, shape=[tgt_len], dtype=tf.int64))
    candidate_matrix = tf.cast(
            tf.logical_and(
                    all_indices[None, :] >= beg_indices[:, None],
                    all_indices[None, :] < end_indices[:, None]),
            tf.float32)
    cumsum_matrix = tf.reshape(
            tf.cumsum(tf.reshape(candidate_matrix, [-1])),
            [-1, tgt_len])
    masked_matrix = tf.cast(cumsum_matrix <= tf.cast(num_predict, dtype=cumsum_matrix.dtype), tf.float32)
    target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
    is_target = tf.cast(target_mask, tf.bool)

    return is_target, target_mask


def _word_span_mask(FLAGS, inputs, tgt_len, num_predict, boundary, stride=1):
    """Sample whole word spans as prediction targets."""
    # Note: 1.2 is roughly the token-to-word ratio
    non_pad_len = tgt_len + 1 - stride
    chunk_len_fp = tf.cast(non_pad_len / num_predict / 1.2, dtype=tf.float32)
    round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

    # Sample span lengths from a zipf distribution
    span_len_seq = np.arange(FLAGS.min_word, FLAGS.max_word + 1)
    probs = np.array([1.0 /  (i + 1) for i in span_len_seq])
    probs /= np.sum(probs)
    logits = tf.constant(np.log(probs), dtype=tf.float32)

    if check_tf_version():
        span_lens = tf.random.categorical(
                logits=logits[None],
                num_samples=num_predict,
                dtype=tf.int64,
        )[0] + FLAGS.min_word
    else:
        span_lens = tf.multinomial(
                logits=logits[None],
                num_samples=num_predict,
                output_dtype=tf.int64,
        )[0] + FLAGS.min_word

    # Sample the ratio [0.0, 1.0) of left context lengths
    span_lens_fp = tf.cast(span_lens, tf.float32)
    left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
    left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)

    left_ctx_len = round_to_int(left_ctx_len)
    right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

    beg_indices = (tf.cumsum(left_ctx_len) +
                                 tf.cumsum(right_offset, exclusive=True))
    end_indices = beg_indices + span_lens

    # Remove out of range `boundary` indices
    max_boundary_index = tf.cast(tf.shape(boundary)[0] - 1, tf.int64)
    valid_idx_mask = end_indices < max_boundary_index
    beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
    end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

    beg_indices = tf.gather(boundary, beg_indices)
    end_indices = tf.gather(boundary, end_indices)

    # Shuffle valid `position` indices
    num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
    order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
    beg_indices = tf.gather(beg_indices, order)
    end_indices = tf.gather(end_indices, order)

    return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                                                     num_predict)


def _token_span_mask(FLAGS, inputs, tgt_len, num_predict, stride=1):
    """Sample token spans as prediction targets."""
    non_pad_len = tgt_len + 1 - stride
    chunk_len_fp = tf.cast(non_pad_len / num_predict, dtype=tf.float32)
    round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

    # Sample span lengths from a zipf distribution
    span_len_seq = np.arange(FLAGS.min_tok, FLAGS.max_tok + 1)
    probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

    probs /= np.sum(probs)
    logits = tf.constant(np.log(probs), dtype=tf.float32)
    if check_tf_version():
        span_lens = tf.random.categorical(
                logits=logits[None],
                num_samples=num_predict,
                dtype=tf.int64,
        )[0] + FLAGS.min_tok
    else:
        span_lens = tf.multinomial(
                logits=logits[None],
                num_samples=num_predict,
                output_dtype=tf.int64,
        )[0] + FLAGS.min_tok

    # Sample the ratio [0.0, 1.0) of left context lengths
    span_lens_fp = tf.cast(span_lens, tf.float32)
    left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
    left_ctx_len = left_ratio * span_lens_fp * (chunk_len_fp - 1)
    left_ctx_len = round_to_int(left_ctx_len)

    # Compute the offset from left start to the right end
    right_offset = round_to_int(span_lens_fp * chunk_len_fp) - left_ctx_len

    # Get the actual begin and end indices
    beg_indices = (tf.cumsum(left_ctx_len) +
                                 tf.cumsum(right_offset, exclusive=True))
    end_indices = beg_indices + span_lens

    # Remove out of range indices
    valid_idx_mask = end_indices < non_pad_len
    beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
    end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

    # Shuffle valid indices
    num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
    order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
    beg_indices = tf.gather(beg_indices, order)
    end_indices = tf.gather(end_indices, order)

    return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                                                     num_predict)


def _whole_word_mask(FLAGS, inputs, tgt_len, num_predict, boundary):
    """Sample whole words as prediction targets."""
    pair_indices = tf.concat([boundary[:-1, None], boundary[1:, None]], axis=1)
    cand_pair_indices = tf.random.shuffle(pair_indices)[:num_predict]
    beg_indices = cand_pair_indices[:, 0]
    end_indices = cand_pair_indices[:, 1]

    return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                                                     num_predict)

def _pmi_mask(FLAGS, inputs, tgt_len, num_predict, 
            start_boundary, end_boundary):
    """Sample whole words as prediction targets."""
    pair_indices = tf.concat([start_boundary[:, None], end_boundary[:, None]], axis=1)
    cand_pair_indices = tf.random.shuffle(pair_indices)[:num_predict]
    beg_indices = cand_pair_indices[:, 0]
    end_indices = cand_pair_indices[:, 1]

    return _idx_pair_to_mask(FLAGS, beg_indices, end_indices, inputs, tgt_len,
                                                     num_predict)


def _single_token_mask(FLAGS, inputs, tgt_len, num_predict, exclude_mask=None):
    """Sample individual tokens as prediction targets."""
    func_mask = tf.equal(inputs, FLAGS.cls_id)
    func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.sep_id))
    func_mask = tf.logical_or(func_mask, tf.equal(inputs, FLAGS.pad_id))
    if exclude_mask is None:
        exclude_mask = func_mask
    else:
        exclude_mask = tf.logical_or(func_mask, exclude_mask)
    candidate_mask = tf.logical_not(exclude_mask)

    all_indices = tf.range(tgt_len, dtype=tf.int64)
    candidate_indices = tf.boolean_mask(all_indices, candidate_mask)
    masked_pos = tf.random.shuffle(candidate_indices)
    if check_tf_version():
        masked_pos = tf.sort(masked_pos[:num_predict])
    else:
        masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
    target_mask = tf.sparse_to_dense(
            sparse_indices=masked_pos,
            output_shape=[tgt_len],
            sparse_values=1.0,
            default_value=0.0)
    is_target = tf.cast(target_mask, tf.bool)

    return is_target, target_mask


def _online_sample_masks(FLAGS, 
        inputs, tgt_len, num_predict, 
        boundary=None, stride=1,
        start_boundary=None,
        end_boundary=None):
    """Sample target positions to predict."""

    # Set the number of tokens to mask out per example
    input_mask = tf.cast(tf.not_equal(inputs, FLAGS.pad_id), dtype=tf.int64)
    num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.float32)
    num_predict = tf.maximum(1, tf.minimum(
            num_predict, tf.cast(tf.round(num_tokens * FLAGS.mask_prob), tf.int32)))
    num_predict = tf.cast(num_predict, tf.int32)

    tf.logging.info("Online sample with strategy: `%s`.", FLAGS.sample_strategy)
    if FLAGS.sample_strategy == "single_token":
        return _single_token_mask(FLAGS, inputs, tgt_len, num_predict)
    else:
        if FLAGS.sample_strategy == "whole_word":
            assert boundary is not None, "whole word sampling requires `boundary`"
            is_target, target_mask = _whole_word_mask(FLAGS, inputs, tgt_len, num_predict,
                                                                                                boundary)
        elif FLAGS.sample_strategy == 'pmi_span':
            assert start_boundary is not None, "whole word sampling requires `boundary`"
            assert end_boundary is not None, "whole word sampling requires `boundary`"
            is_target, target_mask = _pmi_mask(FLAGS, inputs, tgt_len, num_predict,
                                                    boundary, 
                                                    start_boundary=start_boundary,
                                                    end_boundary=end_boundary)
        elif FLAGS.sample_strategy == "token_span":
            is_target, target_mask = _token_span_mask(FLAGS, inputs, tgt_len, num_predict,
                                                                                                stride=stride)
        elif FLAGS.sample_strategy == "word_span":
            assert boundary is not None, "word span sampling requires `boundary`"
            is_target, target_mask = _word_span_mask(FLAGS, inputs, tgt_len, num_predict,
                                                                                             boundary, stride=stride)
        else:
            raise NotImplementedError

        valid_mask = tf.not_equal(inputs, FLAGS.pad_id)
        is_target = tf.logical_and(valid_mask, is_target)
        target_mask = target_mask * tf.cast(valid_mask, tf.float32)

        # Fill in single tokens if not full
        cur_num_masked = tf.reduce_sum(tf.cast(is_target, tf.int32))
        extra_mask, extra_tgt_mask = _single_token_mask(FLAGS,
                inputs, tgt_len, num_predict - cur_num_masked, is_target)
        return tf.logical_or(is_target, extra_mask), target_mask + extra_tgt_mask


def discrepancy_correction(FLAGS, inputs, is_target, tgt_len):
    """Construct the masked input."""
    random_p = tf.random.uniform([tgt_len], maxval=1.0)
    mask_ids = tf.constant(FLAGS.mask_id, dtype=inputs.dtype, shape=[tgt_len])

    change_to_mask = tf.logical_and(random_p > FLAGS.leak_ratio, is_target)
    masked_ids = tf.where(change_to_mask, mask_ids, inputs)

    if FLAGS.rand_ratio > 0:
        change_to_rand = tf.logical_and(
                FLAGS.leak_ratio < random_p,
                random_p < FLAGS.leak_ratio + FLAGS.rand_ratio)
        change_to_rand = tf.logical_and(change_to_rand, is_target)
        rand_ids = tf.random.uniform([tgt_len], maxval=FLAGS.vocab_size,
                                                                 dtype=masked_ids.dtype)
        if FLAGS.get('confusion_matrix', None) is not None and FLAGS.get('confusion_mask_matrix', None) is not None:
            output_ids = confusion_set_sample(rand_ids, 
                                                tgt_len,
                                                FLAGS.confusion_matrix, 
                                                FLAGS.confusion_mask_matrix,
                                                0.3
                                                )
            tf.logging.info("==mixture confusion set sampling==")
        else:
            output_ids = tf.identity(rand_ids)
            tf.logging.info("==random set sampling==")
        masked_ids = tf.where(change_to_rand, output_ids, masked_ids)

    return masked_ids


def create_target_mapping(
        example, is_target, seq_len, num_predict, **kwargs):
    """Create target mapping and retrieve the corresponding kwargs."""
    if num_predict is not None:
        # Get masked indices
        indices = tf.range(seq_len, dtype=tf.int64)
        indices = tf.boolean_mask(indices, is_target)

        # Handle the case that actual_num_predict < num_predict
        actual_num_predict = tf.shape(indices)[0]
        pad_len = num_predict - actual_num_predict

        # Create target mapping
        target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
        paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
        target_mapping = tf.concat([target_mapping, paddings], axis=0)
        example["target_mapping"] = tf.reshape(target_mapping,
                                                                                     [num_predict, seq_len])

        # Handle fields in kwargs
        for k, v in kwargs.items():
            pad_shape = [pad_len] + v.shape.as_list()[1:]
            tgt_shape = [num_predict] + v.shape.as_list()[1:]
            example[k] = tf.concat([
                    tf.boolean_mask(v, is_target),
                    tf.zeros(shape=pad_shape, dtype=v.dtype)], 0)
            example[k].set_shape(tgt_shape)
    else:
        for k, v in kwargs.items():
            example[k] = v

def _decode_record(FLAGS, record, num_predict,
                                    seq_len, 
                                    use_bfloat16=False, 
                                    truncate_seq=False, 
                                    stride=1,
                                    input_type='normal'):
    max_seq_length = seq_len
    if input_type == "normal":
        record_spec = {
                    "input_ori_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "segment_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64)
        }
    elif input_type == "gatedcnn":
        record_spec = {
                    "input_ids_a":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "input_ids_b":
                            tf.FixedLenFeature([max_seq_length], tf.int64)
        }
    elif input_type == 'normal_so':
        record_spec = {
                    "input_ori_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "segment_ids":
                            tf.FixedLenFeature([max_seq_length], tf.int64),
                    "label":tf.FixedLenFeature([], tf.int64)
        }
    if FLAGS.sample_strategy in ["whole_word", "word_span"]:
        tf.logging.info("Add `boundary` spec for %s", FLAGS.sample_strategy)
        record_spec["boundary"] = tf.VarLenFeature(tf.int64)

    example = tf.parse_single_example(record, record_spec)
    if input_type == "normal":
        inputs = example.pop("input_ori_ids")
        label = None
    elif input_type == "gatedcnn":
        inputs = example.pop("input_ids_b")
        label = None
    elif input_type == 'normal_so':
        inputs = example.pop("input_ori_ids")
        label = example.pop("label")
    else:
        label = None
    print(inputs.get_shape(), "==inputs shape==")
    if FLAGS.sample_strategy in ["whole_word", "word_span"]:
        boundary = tf.sparse.to_dense(example.pop("boundary"))
    elif FLAGS.sample_strategy in ['pmi_span']:
        start_boundary = tf.sparse.to_dense(example.pop("start_boundary"))
        end_boundary = tf.sparse.to_dense(example.pop("end_boundary"))
    else:
        boundary = None
        start_boundary = None
        end_boundary = None
    if truncate_seq and stride > 1:
        tf.logging.info("Truncate pretrain sequence with stride %d", stride)
        # seq_len = 8, stride = 2:
        #   [cls 1 2 sep 4 5 6 sep] => [cls 1 2 sep 4 5 sep pad]
        padding = tf.constant([FLAGS.sep_id] + [FLAGS.pad_id] * (stride - 1),
                                                    dtype=inputs.dtype)
        inputs = tf.concat([inputs[:-stride], padding], axis=0)
        if boundary is not None:
            valid_boundary_mask = boundary < seq_len - stride
            boundary = tf.boolean_mask(boundary, valid_boundary_mask)

    is_target, target_mask = _online_sample_masks(FLAGS,
                inputs, seq_len, num_predict, boundary=boundary, 
                stride=stride, start_boundary=start_boundary,
                end_boundary=end_boundary)

    masked_input = discrepancy_correction(FLAGS, inputs, is_target, seq_len)
    masked_input = tf.reshape(masked_input, [max_seq_length])
    is_mask = tf.equal(masked_input, FLAGS.mask_id)
    is_pad = tf.equal(masked_input, FLAGS.pad_id)

    origin_input_mask = tf.equal(inputs, FLAGS.pad_id)
    masked_input *= (1 - tf.cast(origin_input_mask, dtype=tf.int64))

    if label is not None:
        example['label'] = label
        tf.logging.info("== adding sentence order classification ==")

    example["masked_input"] = masked_input
    example["origin_input"] = inputs
    example["is_target"] = tf.cast(is_target, dtype=tf.int64) * (1 - tf.cast(origin_input_mask, dtype=tf.int64))
    # example["input_mask"] = tf.cast(tf.logical_or(is_mask, is_pad), tf.float32)
    # example["pad_mask"] = tf.cast(is_pad, tf.float32)
    input_mask = tf.logical_or(tf.logical_or(is_mask, is_pad), origin_input_mask)
    example["masked_mask"] = 1.0 - tf.cast(tf.logical_or(is_mask, is_pad), dtype=tf.float32)
    pad_mask = tf.logical_or(origin_input_mask, is_pad)
    example["pad_mask"] = 1.0 - tf.cast(pad_mask, tf.float32)

    if FLAGS.get("prepare_text_infilling", False):
        [text_infilling_ids, 
        text_infilling_mask] = prepare_text_infilling(masked_input, duplicate_ids=FLAGS.mask_id)
        text_infilling_mask = tf.cast(text_infilling_mask, tf.float32)
        
        text_infilling_len = tf.reduce_sum(text_infilling_mask * example["pad_mask"], axis=0)
        text_infilling_len = tf.cast(text_infilling_len, dtype=tf.int32)
        text_infilling_ids = tf.boolean_mask(text_infilling_ids, tf.not_equal(text_infilling_ids, 0))

        pad_tensor = tf.zeros((max_seq_length-text_infilling_len), dtype=text_infilling_ids.dtype)
        text_infilling_ids = tf.concat([text_infilling_ids, pad_tensor], axis=0)
        text_infilling_mask_from_ids = tf.cast(tf.not_equal(text_infilling_ids, 0), dtype=tf.float32)

        tgt_shape = inputs.shape.as_list()
        text_infilling_ids.set_shape(tgt_shape)
        text_infilling_mask_from_ids.set_shape(tgt_shape)
        print(text_infilling_ids.get_shape(), 
                    text_infilling_mask_from_ids.get_shape(), 
                "===infilling shape=== vs tgt shape===", 
                tgt_shape)

        example['infilling_pad_mask'] = 1.0 - tf.cast(pad_mask, tf.float32)
        example["pad_mask"] = text_infilling_mask_from_ids
        example['infilled_input'] = text_infilling_ids
        # example["pad_mask"] *= text_infilling_mask
        # example["masked_mask"] = tf.cast(tf.logical_or(tf.cast(example["masked_mask"], tf.bool), 
        #                                                                                           tf.cast(example["pad_mask"], tf.bool)),
        #                                                                   tf.float32)
        # example['infilled_input'] = masked_input
        tf.logging.info("**** prepare text_infilling_ids ****")

    # create target mapping
    create_target_mapping(
            example, is_target, seq_len, num_predict,
            target_mask=target_mask, target=inputs)

    example["masked_lm_positions"] = tf.argmax(example['target_mapping'], axis=-1)
    example["masked_lm_weights"] = example['target_mask']
    example["masked_lm_ids"] = example['target']

    # type cast for example
    convert_example(example, use_bfloat16)

    for k, v in example.items():
        tf.logging.info("%s: %s", k, v)

    return example


def input_fn_builder( 
                     input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads,
                     FLAGS,
                     truncate_seq=False, 
                     use_bfloat16=False,
                     stride=1,
                     input_type="normal"):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    def input_fn(params):
        if FLAGS.get('confusion_matrix', None) is not None and FLAGS.get('confusion_mask_matrix', None) is not None:
            FLAGS.confusion_matrix = tf.constant(FLAGS.confusion_matrix, dtype=tf.int32)
            FLAGS.confusion_mask_matrix = tf.constant(FLAGS.confusion_mask_matrix, dtype=tf.int32)
            tf.logging.info("==convert confusion set to tf-tensor==")

        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                    tf.contrib.data.parallel_interleave(
                            tf.data.TFRecordDataset,
                            sloppy=is_training,
                            cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
                tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(FLAGS, record, 
                                    max_predictions_per_seq,
                                    max_seq_length, 
                                    use_bfloat16=use_bfloat16, 
                                    truncate_seq=truncate_seq, 
                                    stride=stride,
                                    input_type=input_type),
                        batch_size=batch_size,
                        num_parallel_batches=num_cpu_threads,
                        drop_remainder=True))
        return d

    return input_fn


# def get_dataset(
#       params, num_hosts, num_core_per_host, split, file_paths, seq_len,
#       num_predict, use_bfloat16=False, truncate_seq=False, stride=1):
#   """Get one-stream dataset."""
#   #### Function used to parse tfrecord
#   def parser(record):
#       """function used to parse tfrecord."""

#       record_spec = {
#               "input": tf.io.FixedLenFeature([seq_len], tf.int64),
#               "seg_id": tf.io.FixedLenFeature([seq_len], tf.int64)
#       }

#       if FLAGS.sample_strategy in ["whole_word", "word_span"]:
#           tf.logging.info("Add `boundary` spec for %s", FLAGS.sample_strategy)
#           record_spec["boundary"] = tf.io.VarLenFeature(tf.int64)

#       # retrieve serialized example
#       example = tf.io.parse_single_example(
#               serialized=record,
#               features=record_spec)

#       inputs = example.pop("input")
#       if FLAGS.sample_strategy in ["whole_word", "word_span"]:
#           boundary = tf.sparse.to_dense(example.pop("boundary"))
#       else:
#           boundary = None

#       if truncate_seq and stride > 1:
#           tf.logging.info("Truncate pretrain sequence with stride %d", stride)
#           # seq_len = 8, stride = 2:
#           #   [cls 1 2 sep 4 5 6 sep] => [cls 1 2 sep 4 5 sep pad]
#           padding = tf.constant([FLAGS.sep_id] + [FLAGS.pad_id] * (stride - 1),
#                                                       dtype=inputs.dtype)
#           inputs = tf.concat([inputs[:-stride], padding], axis=0)
#           if boundary is not None:
#               valid_boundary_mask = boundary < seq_len - stride
#               boundary = tf.boolean_mask(boundary, valid_boundary_mask)

#       is_target, target_mask = _online_sample_masks(
#               inputs, seq_len, num_predict, boundary=boundary, stride=stride)

#       masked_input = discrepancy_correction(inputs, is_target, seq_len)
#       masked_input = tf.reshape(masked_input, [seq_len])
#       is_mask = tf.equal(masked_input, FLAGS.mask_id)
#       is_pad = tf.equal(masked_input, FLAGS.pad_id)

#       example["masked_input"] = masked_input
#       example["origin_input"] = inputs
#       example["is_target"] = is_target
#       example["input_mask"] = tf.cast(tf.logical_or(is_mask, is_pad), tf.float32)
#       example["pad_mask"] = tf.cast(is_pad, tf.float32)

#       # create target mapping
#       create_target_mapping(
#               example, is_target, seq_len, num_predict,
#               target_mask=target_mask, target=inputs)

#       # type cast for example
#       data_utils.convert_example(example, use_bfloat16)

#       for k, v in example.items():
#           tf.logging.info("%s: %s", k, v)

#       return example

#   # Get dataset
#   dataset = parse_files_to_dataset(
#           parser=parser,
#           file_paths=file_paths,
#           split=split,
#           num_hosts=num_hosts,
#           host_id=host_id,
#           num_core_per_host=num_core_per_host,
#           bsz_per_core=bsz_per_core)

#   return dataset


# def get_input_fn(
#       tfrecord_dir,
#       split,
#       bsz_per_host,
#       seq_len,
#       num_predict,
#       num_hosts=1,
#       num_core_per_host=1,
#       uncased=False,
#       num_passes=None,
#       use_bfloat16=False,
#       num_pool=0,
#       truncate_seq=False):
#   """Create Estimator input function."""

#   assert num_predict is not None and 0 < num_predict < seq_len - 3
#   stride = 2 ** num_pool

#   # Merge all record infos into a single one
#   record_glob_base = data_utils.format_filename(
#           prefix="meta.{}.pass-*".format(split),
#           suffix="json*",
#           seq_len=seq_len,
#           uncased=uncased)

#   def _get_num_batch(info):
#       if "num_batch" in info:
#           return info["num_batch"]
#       elif "num_example" in info:
#           return info["num_example"] / bsz_per_host
#       else:
#           raise ValueError("Do not have sample info.")

#   record_info = {"num_batch": 0, "filenames": []}

#   tfrecord_dirs = tfrecord_dir.split(",")
#   tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

#   for idx, record_dir in enumerate(tfrecord_dirs):
#       record_glob = os.path.join(record_dir, record_glob_base)
#       tf.logging.info("[%d] Record glob: %s", idx, record_glob)

#       record_paths = sorted(tf.io.gfile.glob(record_glob))
#       tf.logging.info("[%d] Num of record info path: %d",
#                                       idx, len(record_paths))

#       cur_record_info = {"num_batch": 0, "filenames": []}

#       for record_info_path in record_paths:
#           if num_passes is not None:
#               record_info_name = os.path.basename(record_info_path)
#               fields = record_info_name.split(".")[2].split("-")
#               pass_id = int(fields[-1])
#               if pass_id >= num_passes:
#                   tf.logging.debug("Skip pass %d: %s", pass_id, record_info_name)
#                   continue

#           with tf.io.gfile.GFile(record_info_path, "r") as fp:
#               info = json.load(fp)
#               cur_record_info["num_batch"] += int(_get_num_batch(info))
#               cur_record_info["filenames"] += info["filenames"]

#       # overwrite directory for `cur_record_info`
#       new_filenames = []
#       for filename in cur_record_info["filenames"]:
#           basename = os.path.basename(filename)
#           new_filename = os.path.join(record_dir, basename)
#           new_filenames.append(new_filename)
#       cur_record_info["filenames"] = new_filenames

#       tf.logging.info("[Dir %d] Number of chosen batches: %s",
#                                       idx, cur_record_info["num_batch"])
#       tf.logging.info("[Dir %d] Number of chosen files: %s",
#                                       idx, len(cur_record_info["filenames"]))
#       tf.logging.debug(cur_record_info["filenames"])

#       # add `cur_record_info` to global `record_info`
#       record_info["num_batch"] += cur_record_info["num_batch"]
#       record_info["filenames"] += cur_record_info["filenames"]

#   tf.logging.info("Total number of batches: %d", record_info["num_batch"])
#   tf.logging.info("Total number of files: %d", len(record_info["filenames"]))
#   tf.logging.debug(record_info["filenames"])

#   def input_fn(params):
#       """Input function wrapper."""
#       dataset = get_dataset(
#               params=params,
#               num_hosts=num_hosts,
#               num_core_per_host=num_core_per_host,
#               split=split,
#               file_paths=record_info["filenames"],
#               seq_len=seq_len,
#               use_bfloat16=use_bfloat16,
#               num_predict=num_predict,
#               truncate_seq=truncate_seq,
#               stride=stride)

#       return dataset

#   return input_fn, record_info
