from tensorflow.python.layers import core as layers_core
import time
import os
import itertools
import pickle
import numpy as np
import tensorflow as tf

from utils import utils
from utils import data_utils


class Chatbot(object):

    def __init__(self, params, language_base, sess):

        self.params = params

        with tf.variable_scope("root"):
            self.train_model = ChatModel(params, tf.contrib.learn.ModeKeys.TRAIN, language_base, sess)
        with tf.variable_scope('root', reuse=True):
            self.eval_model = ChatModel(params, tf.contrib.learn.ModeKeys.EVAL, language_base, sess)
        with tf.variable_scope('root', reuse=True):
            self.infer_model = ChatModel(params, tf.contrib.learn.ModeKeys.INFER, language_base, sess)

    def train(self, data, sess):

        return self.train_model.train(data, sess)

    def infer(self, data, sess):

        return self.infer_model.infer(data, sess)

    def eval(self, data, sess):

        return self.eval_model.eval(data, sess)

    def compute_perplexity(self, data_generator, sess):

        total_loss = 0
        total_predict_count = 0

        while True:
            try:
                batch_data = next(data_generator)
                loss, predict_count = self.eval(batch_data, sess)
                total_loss += loss * self.params['batch_size']
                total_predict_count += predict_count
            except StopIteration:
                break

        perplexity = utils.safe_exp(total_loss / total_predict_count)

        return perplexity

    def save(self, directory):

        with open(os.path.join(directory, 'params'), 'wb') as pickle_file:
            pickle.dump(self.params, pickle_file)

    @classmethod
    def load(cls, directory, language_base):

        with open(os.path.join(directory, 'params'), 'rb') as pickle_file:
            params = pickle.load(pickle_file)

        return cls(params, language_base, session)


class ChatModel(object):

    def __init__(self, params, mode, language_base, sess):

        self.mode = mode

        # Placeholders
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(params['src_max_len'], params['batch_size']), name='encoder_inputs')
        self.encoder_input_lengths = tf.placeholder(dtype=tf.int32, shape=(params['batch_size'],), name='encoder_input_lengths')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(params['tgt_max_len'], params['batch_size']), name='decoder_inputs')
        self.decoder_input_lengths = tf.placeholder(dtype=tf.int32, shape=(params['batch_size'],), name='decoder_input_lengths')
        self.decoder_outputs = tf.placeholder(dtype=tf.int32, shape=(params['tgt_max_len'], params['batch_size']), name='decoder_outputs')

        # Projection layer
        self.output_layer = layers_core.Dense(len(language_base.vocabulary), use_bias=False, name='output_projection')

        # Look up embedding:
        #   encoder_inputs: [max_time, batch_size]
        #   encoder_emb_inp: [max_time, batch_size, embedding_size]
        encoder_emb_inp = tf.nn.embedding_lookup(language_base.embeddings, self.encoder_inputs)
        decoder_emb_inp = tf.nn.embedding_lookup(language_base.embeddings, self.decoder_inputs)

        encoder_emb_inp.set_shape([params['src_max_len'], params['batch_size'], language_base.word_embedding_size])
        decoder_emb_inp.set_shape([params['tgt_max_len'], params['batch_size'], language_base.word_embedding_size])

        params['num_encoder_layers'] = int(params['num_layers'] / 2)
        params['num_decoder_units'] = params['num_units']

        # Build encoder
        encoder_outputs, encoder_state = self._build_bidirectional_encoder(encoder_emb_inp, params)

        # Build decoder
        logits, self.translations = self._build_decoder(
            encoder_state, encoder_outputs, decoder_emb_inp,
            language_base, params)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:

            self.predict_count = tf.reduce_sum(self.decoder_input_lengths)
            self.loss = self._compute_loss(logits, params)

            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])

            self.update_step = self._build_optimizer(clipped_gradients, trainable_variables, params)

    def _build_bidirectional_encoder(self, encoder_emb_inp, params):

        # Build RNN cell
        encoder_fw_cell = self._build_cell(params['num_units'], params['num_encoder_layers'])
        encoder_bw_cell = self._build_cell(params['num_units'], params['num_encoder_layers'])

        bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_fw_cell, encoder_bw_cell, encoder_emb_inp,
            sequence_length=self.encoder_input_lengths, dtype=tf.float32, time_major=True)

        encoder_outputs = tf.concat(bi_encoder_outputs, -1)

        if params['num_encoder_layers'] == 1:
            encoder_state = bi_encoder_state
        else:
            encoder_state = []
            for layer_id in range(params['num_encoder_layers']):
                encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                encoder_state.append(bi_encoder_state[1][layer_id])  # backward
            encoder_state = tuple(encoder_state)

        return encoder_outputs, encoder_state

    def _build_cell(self, num_units, num_layers):

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(num_units)

        if num_layers == 1:
            return lstm_cell()
        # return lstm_cell()
        return tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

    def _build_optimizer(self, clipped_gradients, trainable_variables, params):

        # Optimization
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        update_step = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables))

        return update_step

    def _build_decoder(self, encoder_state, encoder_outputs, decoder_emb_inp, language_base, params):

        attention_mechanism = self._luong_attention_mechanism(encoder_outputs, params)

        self.decoder_cell = self._build_cell(params['num_decoder_units'], params['num_layers'])
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism,
            attention_layer_size=params['num_decoder_units'])

        batch_size = params['batch_size']

        if self.mode == tf.contrib.learn.ModeKeys.INFER:

            if params['beam_width'] > 0:
                encoder_state = tf.contrib.seq2seq.tile_batch(
                    encoder_state, multiplier=params['beam_width'])
                batch_size = params['batch_size'] * params['beam_width']

            self.decoder_initial_state = self.decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            self.maximum_iterations = tf.round(tf.reduce_max(self.encoder_input_lengths) * 2)

            tmp_embeddings = tf.identity(language_base.embeddings)
            tmp_embeddings.set_shape([len(language_base.vocabulary), language_base.word_embedding_size])

            # # Helper
            # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            #     tmp_embeddings, tf.fill([params.batch_size], sos_id), eos_id)
            #
            # # Decoder
            # decoder = tf.contrib.seq2seq.BasicDecoder(
            #     decoder_cell, helper, decoder_initial_state,
            #     output_layer=self.output_layer)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=tmp_embeddings,
                start_tokens=tf.fill([params['batch_size']], language_base.sos_id),
                end_token=language_base.eos_id,
                initial_state=self.decoder_initial_state,
                beam_width=params['beam_width'],
                output_layer=self.output_layer,
                length_penalty_weight=params['length_penalty_weight'])

            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.maximum_iterations, output_time_major=True)

            logits = None
            sample_id = outputs.predicted_ids
            sample_id = tf.transpose(sample_id, perm=[1, 2, 0])

        else:

            decoder_initial_state = self.decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, self.decoder_input_lengths,
                time_major=True)

            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, helper, decoder_initial_state,
                output_layer=self.output_layer)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True)

            # logits = self.output_layer(self.outputs.rnn_output)
            logits = outputs.rnn_output
            sample_id = outputs.sample_id

        return logits, sample_id

    def _compute_loss(self, logits, params):

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.decoder_outputs, logits=logits)

        target_weights = tf.sequence_mask(
            self.decoder_input_lengths, params['tgt_max_len'],
            dtype=logits.dtype)

        target_weights = tf.transpose(target_weights)
        loss = (tf.reduce_sum(crossent * target_weights) / tf.to_float(params['batch_size']))

        return loss

    def eval(self, batch, sess):

        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.loss, self.predict_count], feed_dict=self._get_feed_dict(batch))

    def train(self, batch, sess):

        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update_step, self.loss, self.predict_count], feed_dict=self._get_feed_dict(batch))

    def infer(self, data, sess):

        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.translations], feed_dict=self._get_feed_dict(data))

    def _get_feed_dict(self, data):

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            return {
                self.encoder_inputs: data[0],
                self.encoder_input_lengths: data[3]
            }
        else:
            return {
                self.encoder_inputs: data[0],
                self.encoder_input_lengths: data[3],
                self.decoder_inputs: data[1],
                self.decoder_input_lengths: data[4],
                self.decoder_outputs: data[2],
            }

    def _luong_attention_mechanism(self, memory, params):

        # attention_states: [batch_size, max_time, num_units]
        memory = tf.transpose(memory, [1, 0, 2])
        source_sequence_length = self.encoder_input_lengths

        if self.mode == tf.contrib.learn.ModeKeys.INFER and params['beam_width'] > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=params['beam_width'])
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=params['beam_width'])

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            params['num_decoder_units'], memory,
            memory_sequence_length=source_sequence_length)

        return attention_mechanism
