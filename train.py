from nltk.tokenize import word_tokenize
from tensorflow.python.layers import core as layers_core
from nltk.translate.bleu_score import sentence_bleu
import os
import itertools
import random
import time
import numpy as np
import tensorflow as tf
tf.reset_default_graph()

from core.language import LanguageBase
from model import Chatbot
from utils import data_utils
from utils import utils
from params import chat_params

if __name__ == '__main__':

    session = tf.InteractiveSession()

    # Load data
    src_train, tgt_train = data_utils.load_data_from_file('train', max_data=chat_params['max_data'])

    # Create datasets
    ingestion_data = src_train + tgt_train

    # Initialize language base
    language_base = LanguageBase()

    session.run(tf.global_variables_initializer())

    # Ingest data into language base
    language_base.ingest(ingestion_data, session, batch_size=50)
    chatbot = Chatbot(chat_params, language_base, session)
    session.run(tf.variables_initializer([x for x in tf.global_variables() if x not in [language_base.embeddings,
                                                                                        language_base.nce_weights,
                                                                                        language_base.nce_biases]]))

    # Create data generator
    train_data_generator = data_utils.create_data_generator(src_train,
                                                            tgt_train,
                                                            batch_size=chat_params['batch_size'],
                                                            src_max_len=chat_params['src_max_len'],
                                                            tgt_max_len=chat_params['tgt_max_len'],
                                                            language_base=language_base)

    global_step = 0
    loss_track = []

    saver = tf.train.Saver()

    train_data_generator, first_dev_generator = itertools.tee(train_data_generator)
    first_dev_ppl = chatbot.compute_perplexity(first_dev_generator, session)
    train_data_generator, first_test_generator = itertools.tee(train_data_generator)
    first_test_ppl = chatbot.compute_perplexity(first_test_generator, session)

    print("# First evaluation, global step 0")
    print("  eval dev: perplexity {0:.2f}".format(first_dev_ppl))
    print("  eval test: perplexity {0:.2f}".format(first_test_ppl))

    for i in range(chat_params['n_epochs']):
        print("# Start epoch {}, step {}".format(i+1, global_step))
        train_data_generator, epoch_train_generator = itertools.tee(train_data_generator)
        while True:
            try:
                step_start_time = time.time()
                batch_data = next(epoch_train_generator)
                _, loss, predict_count = chatbot.train(batch_data, session)
                global_step += chat_params['batch_size']
                loss_track.append(loss)

                if global_step % chat_params['steps_per_log'] == 0:
                    ppl = utils.safe_exp((loss * chat_params['batch_size']) / predict_count)
                    print('  epoch {0} step {1} loss {2:.2f} step-time {3:.2f} ppl {4:.2f}'.format(
                        i+1, global_step, loss, time.time() - step_start_time, ppl))

                if global_step % chat_params['steps_per_checkpoint'] == 0:
                    translations = chatbot.infer(batch_data, session)

                    batched_encoder_inputs = data_utils.convert_to_batch_major(batch_data[0])
                    batched_decoder_outputs = data_utils.convert_to_batch_major(batch_data[2])

                    src_words = data_utils.trim_words([language_base.reversed_vocabulary[w] for w in batched_encoder_inputs[0]])
                    tgt_words = data_utils.trim_words([language_base.reversed_vocabulary[w] for w in batched_decoder_outputs[0]])
                    predicted_words = data_utils.trim_words([language_base.reversed_vocabulary[w] for w in translations[0][0][0]])

                    src_text = data_utils.words_to_text(src_words)
                    tgt_text = data_utils.words_to_text(tgt_words)
                    predicted_text = data_utils.words_to_text(predicted_words)

                    bleu = sentence_bleu([tgt_words], predicted_words)

                    print("  CHECKPOINT EVAL, bleu: {0:.2f}".format(bleu))
                    print("    src:", src_text)
                    print("    tgt:", tgt_text)
                    print("    predicted:", predicted_text)

                    saver.save(session, os.path.join(chat_params['model_dir'], 'model.ckpt'))

            except StopIteration:
                break

    chatbot.save(chat_params['model_dir'])
    language_base.save(chat_params['language_base_dir'])
