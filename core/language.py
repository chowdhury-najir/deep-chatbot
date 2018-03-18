from nltk.tokenize import word_tokenize
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import time
import itertools
import math
import tensorflow as tf
import numpy as np

class LanguageBase(object):

    def __init__(self,
                 directory,
                 word_embedding_size=300,
                 learning_rate=1.0,
                 session=None):

        self.directory = directory
        self.word_embedding_size = word_embedding_size
        self.learning_rate = learning_rate

        self.session = session
        if self.session is None:
            self.session = tf.InteractiveSession()

        self.vocabulary = {'<pad>': 0, '<s>': 1, '</s>': 2}
        self.reversed_vocabulary = ['<pad>', '<s>', '</s>']
        self.weighted_word_multiplier = TfidfVectorizer(tokenizer=word_tokenize)

        self.embeddings = tf.Variable(tf.random_uniform([len(self.vocabulary), self.word_embedding_size], -1.0, 1.0),
                                      name='embeddings')
        self.nce_weights = tf.Variable(tf.truncated_normal([len(self.vocabulary), self.word_embedding_size],
                                                           stddev=1.0 / math.sqrt(self.word_embedding_size)),
                                       name='nce_weights')
        self.nce_biases = tf.Variable(tf.zeros([len(self.vocabulary)]), name='nce_biases')

        self.session.run(tf.variables_initializer([self.embeddings, self.nce_weights, self.nce_biases]))

    @staticmethod
    def __create_features(documents, context_window_size=3):

        word_tuples = []

        for document in documents:
            tuples = []
            words = word_tokenize(document.lower())
            padding = ['<pad>' for _ in range(context_window_size)]

            words = padding + words + padding

            for i in range(context_window_size, len(words) - context_window_size):
                for j in range(1, context_window_size + 1):
                    tuples.append((words[i], words[i-j]))
                    tuples.append((words[i], words[i+j]))

            word_tuples += tuples

        return word_tuples

    def update_vocabulary(self, words):

        new_words = 0
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)
                self.reversed_vocabulary.append(word)
                new_words += 1

        self.embeddings = tf.Variable(tf.concat([self.embeddings, tf.random_uniform([new_words, self.word_embedding_size], -1.0, 1.0)], 0), name="embeddings")
        self.nce_weights = tf.Variable(tf.concat([self.nce_weights, tf.truncated_normal([new_words, self.word_embedding_size], stddev=1.0 / math.sqrt(self.word_embedding_size))], 0), name='nce_weights')
        self.nce_biases = tf.Variable(tf.concat([self.nce_biases, tf.zeros([new_words])], 0), name='nce_biases')

        self.session.run(tf.variables_initializer([self.embeddings, self.nce_weights, self.nce_biases]))

    def __save_word_embeddings_metadata(self):

        with open(os.path.join(self.directory, 'metadata.tsv'), 'w') as f:
            f.write('ID\tWord\n')
            for w in self.vocabulary:
                f.write(str(self.vocabulary[w]) + '\t' + str(w) + '\n')

    def __update_embedding_projector(self):

        self.__save_word_embeddings_metadata()

        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(self.directory)

        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)

    def convert_words_to_ids(self, words):

        return [self.vocabulary[w] for w in words]

    def calculate_document_vector(self, document, use_weights=True):

        word_vectors = self.get_word_vectors(document, use_weights=use_weights)

        return np.array(np.mean(word_vectors, axis=0))

    def get_word_vectors(self, text, use_weights=False):

        word_vectors = []

        words = word_tokenize(text.lower())

        for w in words:
            word_vectors.append(tf.nn.embedding_lookup(self.embeddings, self.vocabulary[w]).eval(session=self.session))

        if use_weights:
            word_weights = dict(zip(self.weighted_word_multiplier.get_feature_names(), self.weighted_word_multiplier.idf_))

            for i in range(len(words)):
                word_vectors[i] = word_vectors[i] * word_weights[words[i]]

        return word_vectors

    @staticmethod
    def load(directory):

        tf.reset_default_graph()

        with open(os.path.join(directory, 'language_base_obj.pkl'), 'rb') as pickle_file:
            obj = pickle.load(pickle_file)

        obj.embeddings = tf.get_variable('embeddings', shape=(len(obj.vocabulary), obj.word_embedding_size))
        obj.nce_weights = tf.get_variable('nce_weights', shape=(len(obj.vocabulary), obj.word_embedding_size))
        obj.nce_biases = tf.get_variable('nce_biases', shape=(len(obj.vocabulary),))

        saver = tf.train.Saver()
        obj.session = tf.InteractiveSession()
        saver.restore(obj.session, os.path.join(directory, 'model.ckpt'))

        return obj

    def __save_session(self):

        saver = tf.train.Saver()
        saver.save(self.session, os.path.join(self.directory, 'model.ckpt'))

    def save(self):

        self.__update_embedding_projector()
        self.__save_session()

        self.session = None
        self.embeddings = None
        self.nce_weights = None
        self.nce_biases = None

        with open(os.path.join(self.directory, 'language_base_obj.pkl'), 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def ingest(self,
               documents,
               num_sampled=64,
               batch_size=10,
               n_iter=1,
               context_window_size=3,
               steps_per_checkpoint=None,
               steps_per_projection_update=None,
               verbose=1):

        if not isinstance(documents, list):
            batch_size = 1
            documents = [documents]

        if not hasattr(self.weighted_word_multiplier, 'vocabulary_'):
            self.weighted_word_multiplier.fit(documents)
            self.weighted_word_multiplier.n_docs = len(documents)
        else:
            self.weighted_word_multiplier.partial_fit(documents)

        word_tuples = LanguageBase.__create_features(documents, context_window_size)

        words = [x[0] for x in word_tuples] + [x[1] for x in word_tuples]
        self.update_vocabulary(words)

        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                             biases=self.nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=len(self.vocabulary)))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        step = 0
        prev_step = 0
        start_time = time.time()
        _loss = 0

        for _ in range(n_iter):

            data = word_tuples[:]

            while len(data) >= batch_size:
                batch = data[:batch_size]
                data = data[batch_size:]

                batch = [(self.vocabulary[x[0]], self.vocabulary[x[1]]) for x in batch]

                inputs = np.array([x[0] for x in batch])
                labels = np.array([x[1] for x in batch]).reshape(-1, 1)
                feed_dict = {
                    train_inputs: inputs,
                    train_labels: labels
                }
                _, cur_loss = self.session.run([optimizer, loss], feed_dict=feed_dict)
                _loss = cur_loss
                step += batch_size
                if steps_per_projection_update is not None:
                    for i in range(prev_step, step):
                        if i % steps_per_projection_update == 0 and i != 0 or steps_per_projection_update == 1:
                            self.__update_embedding_projector()
                if steps_per_checkpoint is not None:
                    for i in range(prev_step, step):
                        if i % steps_per_checkpoint == 0 and i != 0:
                            self.__save_session()
                            if verbose:
                                print('step: {}, loss: {}, time: {}'.format(i, cur_loss, time.time() - start_time))
                prev_step = step

            self.__update_embedding_projector()
            self.__save_session()

            if verbose:
                print('end of epoch. step: {}, loss: {}, time: {}'.format(step, _loss, time.time() - start_time))

        return True

    def word_embeddings(self):

        return self.embeddings.eval()
