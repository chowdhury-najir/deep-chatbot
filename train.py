from nltk.tokenize import word_tokenize
from tensorflow.python.layers import core as layers_core
import os
import itertools
import random
import time
import numpy as np
import tensorflow as tf
tf.reset_default_graph()

from core.language import LanguageBase
from utils import data_utils
from utils import utils
import params

if __name__ == '__main__':

    session = tf.InteractiveSession()

    # Placeholders
    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(params.src_max_len, params.batch_size), name='encoder_inputs')
    encoder_input_lengths = tf.placeholder(dtype=tf.int32, shape=(params.batch_size,), name='encoder_input_lengths')
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(params.tgt_max_len, params.batch_size), name='decoder_inputs')
    decoder_input_lengths = tf.placeholder(dtype=tf.int32, shape=(params.batch_size,), name='decoder_input_lengths')
    decoder_outputs = tf.placeholder(dtype=tf.int32, shape=(params.tgt_max_len, params.batch_size), name='decoder_outputs')

    # Load data
    src_train, tgt_train = data_utils.load_data_from_file('train', max_data=params.max_data)

    # Create datasets
    ingestion_data = src_train + tgt_train

    # Initialize language base
    language_base = LanguageBase(params.language_base_dir)

    # Ingest data into language base
    language_base.ingest(ingestion_data, batch_size=50)

    # Create data generator
    train_data_generator = data_utils.create_data_generator(src_train,
                                                            tgt_train,
                                                            batch_size=params.batch_size,
                                                            src_max_len=params.src_max_len,
                                                            tgt_max_len=params.tgt_max_len,
                                                            language_base=language_base)

    # Look up embedding:
    #   encoder_inputs: [max_time, batch_size]
    #   encoder_emb_inp: [max_time, batch_size, embedding_size]
    encoder_emb_inp = tf.nn.embedding_lookup(language_base.embeddings, encoder_inputs)
    decoder_emb_inp = tf.nn.embedding_lookup(language_base.embeddings, decoder_inputs)

    sos_id = language_base.vocabulary['<s>']
    eos_id = language_base.vocabulary['</s>']

    # Build RNN cell
    encoder_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_units)
    encoder_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_units)

    # # Run Dynamic RNN
    # #   encoder_outputs: [max_time, batch_size, num_units]
    # #   encoder_state: [batch_size, num_units]
    # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    #     encoder_cell, encoder_emb_inp,
    #     sequence_length=encoder_input_lengths, dtype=tf.float32, time_major=True)

    (bi_encoder_outputs, (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
        encoder_forward_cell, encoder_backward_cell, encoder_emb_inp,
        sequence_length=encoder_input_lengths, dtype=tf.float32, time_major=True)

    encoder_outputs = tf.concat(bi_encoder_outputs, -1)

    encoder_state_c = tf.concat(
        (encoder_fw_state.c, encoder_bw_state.c), 1)

    encoder_state_h = tf.concat(
        (encoder_fw_state.h, encoder_bw_state.h), 1)

    #TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
    encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
        c=encoder_state_c,
        h=encoder_state_h
    )

    # Build RNN cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_units * 2)

    # attention_states: [batch_size, max_time, num_units]
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        params.num_units * 2, attention_states,
        memory_sequence_length=encoder_input_lengths)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=params.num_units * 2)

    # Helpers
    train_helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, decoder_input_lengths, time_major=True)
    eval_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        language_base.embeddings, tf.fill([params.batch_size], sos_id), eos_id
    )

    # decoder_initial_state = encoder_state
    decoder_initial_state = decoder_cell.zero_state(params.batch_size, tf.float32).clone(cell_state=encoder_state)

    # Projection layer
    projection_layer = layers_core.Dense(
        len(language_base.vocabulary), use_bias=False)

    # Decoder
    train_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, train_helper, decoder_initial_state,
        output_layer=projection_layer)
    eval_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, eval_helper, decoder_initial_state,
        output_layer=projection_layer
    )

    maximum_iterations = tf.round(tf.reduce_max(encoder_input_lengths) * 2)

    # Dynamic decoding
    train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
    eval_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        eval_decoder, maximum_iterations=maximum_iterations)

    translations = eval_outputs.sample_id
    train_logits = train_outputs.rnn_output

    train_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_outputs, logits=train_logits)

    # Target weights
    target_weights = tf.sequence_mask(
        decoder_input_lengths, params.tgt_max_len, dtype=train_logits.dtype)
    target_weights = tf.transpose(target_weights)

    # Loss function
    train_loss = (tf.reduce_sum(train_crossent * target_weights) /
        tf.to_float(params.batch_size))

    # Calculate and clip gradients
    train_vars = tf.trainable_variables()
    gradients = tf.gradients(train_loss, train_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, params.max_gradient_norm)

    # Optimization
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, train_vars))

    global_step = 0
    start_time = time.time()
    loss_track = []

    saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())

    for i in range(params.n_epochs):

        train_data_generator, epoch_train_generator = itertools.tee(train_data_generator)

        while True:
            try:

                batch_data = next(epoch_train_generator)

                _, loss = session.run([update_step, train_loss], feed_dict={
                    encoder_inputs: batch_data[0],
                    encoder_input_lengths: batch_data[3],
                    decoder_inputs: batch_data[1],
                    decoder_input_lengths: batch_data[4],
                    decoder_outputs: batch_data[2]
                })

                global_step += params.batch_size
                loss_track.append(loss)

                if global_step % params.steps_per_log == 0:
                    print('epoch: {}, step: {}, loss: {}, time: {}'.format(i+1, global_step, loss, time.time() - start_time))

                if global_step % params.steps_per_checkpoint == 0:

                    res = session.run([translations], feed_dict={
                        encoder_inputs: batch_data[0],
                        encoder_input_lengths: batch_data[3]
                    })

                    src_batch_major = data_utils.convert_to_batch_major(batch_data[0])
                    tgt_batch_major = data_utils.convert_to_batch_major(batch_data[2])

                    src_text = data_utils.words_to_text([language_base.reversed_vocabulary[x] for x in src_batch_major[0]])
                    tgt_text = data_utils.words_to_text([language_base.reversed_vocabulary[x] for x in tgt_batch_major[0]])
                    predicted_text = data_utils.words_to_text([language_base.reversed_vocabulary[x] for x in res[0][0]])

                    print('\neval:')
                    print('\tsrc:', src_text)
                    print('\ttgt:', tgt_text)
                    print('\tpredicted:', predicted_text)

                    print('\nsaving checkpoint to', params.model_dir,'\n')

                    saver.save(session, os.path.join(params.model_dir, 'model.ckpt'))

            except StopIteration:
                break
