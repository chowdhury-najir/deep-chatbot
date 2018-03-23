from nltk.tokenize import word_tokenize
import os
import random
import collections
import itertools
import numpy as np
import tensorflow as tf

import params
from . import utils


def load_data_from_file(file_prefix, max_data=None):

    src_data = []
    tgt_data = []

    with open(os.path.join(params.data_dir, file_prefix + params.src_suffix), 'r') as f:
        src_lines = f.readlines()

    with open(os.path.join(params.data_dir, file_prefix + params.tgt_suffix), 'r') as f:
        tgt_lines = f.readlines()

    if max_data is not None:
        src_lines = src_lines[:max_data]
        tgt_lines = tgt_lines[:max_data]

    for i in range(len(src_lines)):

        src_line = src_lines[i].replace('\n', '').strip()
        tgt_line = tgt_lines[i].replace('\n', '').strip()

        src_data.append(src_line)
        tgt_data.append(tgt_line)

    return src_data, tgt_data


def create_data_generator(src_data, tgt_data, batch_size, src_max_len, tgt_max_len, language_base):

    i = 0
    while i < len(src_data) and i < len(tgt_data):

        batch_src = src_data[i:i+batch_size]
        batch_tgt = tgt_data[i:i+batch_size]

        i += batch_size

        src_words = [word_tokenize(src.lower()) for src in batch_src]
        tgt_input_words = [['<s>'] + word_tokenize(tgt.lower()) for tgt in batch_tgt]
        tgt_output_words = [word_tokenize(tgt.lower()) + ['</s>'] for tgt in batch_tgt]

        padded_src_words = [utils.pad_or_trim_sequence(seq, src_max_len, '<pad>') for seq in src_words]
        padded_tgt_input_words = [utils.pad_or_trim_sequence(seq, tgt_max_len, '<pad>') for seq in tgt_input_words]
        padded_tgt_output_words = [utils.pad_or_trim_sequence(seq, tgt_max_len, '<pad>', True) for seq in tgt_output_words]

        src_ids = [language_base.convert_words_to_ids(seq) for seq in padded_src_words]
        tgt_input_ids = [language_base.convert_words_to_ids(seq) for seq in padded_tgt_input_words]
        tgt_output_ids = [language_base.convert_words_to_ids(seq) for seq in padded_tgt_output_words]

        src_ids, src_lengths = convert_to_time_major(src_ids, src_max_len)
        tgt_input_ids, tgt_lengths = convert_to_time_major(tgt_input_ids, tgt_max_len)
        tgt_output_ids, _ = convert_to_time_major(tgt_output_ids, tgt_max_len)

        yield src_ids, tgt_input_ids, tgt_output_ids, src_lengths, tgt_lengths


def convert_to_batch_major(inputs):

    return inputs.swapaxes(0, 1)


def convert_to_time_major(inputs, max_len):

    batch_size = len(inputs)
    input_seq_lengths = np.array([len(x) for x in inputs])

    inputs_batch_major = np.zeros(shape=[batch_size, max_len], dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, id in enumerate(seq):
            inputs_batch_major[i, j] = id

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, input_seq_lengths


def words_to_text(words):

    for i in range(len(words)):
        if words[i] == '</s>' or words[i] == '<pad>':
            return ' '.join(words[:i])

    return ' '.join(words)
