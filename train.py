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
from model import Chatbot
from utils import data_utils
from utils import utils
import params

if __name__ == '__main__':

    session = tf.InteractiveSession()

    # Load data
    src_train, tgt_train = data_utils.load_data_from_file('train', max_data=params.max_data)

    # Create datasets
    ingestion_data = src_train + tgt_train

    # Initialize language base
    language_base = LanguageBase(params.language_base_dir, session=session)

    chatbot = Chatbot(params, language_base, session)

    session.run(tf.global_variables_initializer())

    # Ingest data into language base
    language_base.ingest(ingestion_data, batch_size=50)

    # Create data generator
    train_data_generator = data_utils.create_data_generator(src_train,
                                                            tgt_train,
                                                            batch_size=params.batch_size,
                                                            src_max_len=params.src_max_len,
                                                            tgt_max_len=params.tgt_max_len,
                                                            language_base=language_base)

    chatbot.train(train_data_generator, session)
