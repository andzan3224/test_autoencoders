import os
import pickle
import random

import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense
from theano import config as TC

#from Utils.checkpointer import load_checkpoint
#from Utils.denoisingautoencoder import DenoisingAutoEncoder, ConvDaeMultiSoftmax

char_to_int = {'q': 0, 'a': 1, 'ą': 2, 'z': 3, 'ż': 4, 'w': 5, 's': 6, 'ś': 7, 'x': 8, 'ź': 9, 'e': 10,
               'ę': 11, 'd': 12, 'c': 13, 'ć': 14, 'r': 15, 'f': 16, 'v': 17, 't': 18, 'g': 19, 'b': 20,
               'y': 21, ' ': 22, 'h': 23, 'n': 24, 'ń': 25, 'u': 26, 'j': 27, 'm': 28, 'i': 29, 'k': 30,
               'o': 31, 'ó': 32, 'l': 33, 'ł': 34, 'p': 35}
int_to_char = u'qaązżwsśxźeędcćrfvtgby hnńujmikoólłp'
nb_chars = len(int_to_char)
max_word_length = 16
word_vector_length = max_word_length * nb_chars


def load_data(filename, number=0, create_set=False):
    file = open(filename, 'r')
    words = file.read().splitlines()
    if number > 0:
        words = words[:number]
    word_set = set(words) if create_set else None
    return words, word_set


def maybe_pickle(file_path, number_of_words=0, force=False):
    pickle_file_path = file_path + '.pkl'
    word_list, word_set = None, None
    if os.path.exists(pickle_file_path) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping pickling. Loading.' % pickle_file_path)
        try:
            with open(pickle_file_path, 'rb') as f:
                word_list = pickle.load(f)
                word_set = pickle.load(f)
        except Exception as e:
            print('Unable to load data from', pickle_file_path, ':', e)
    else:
        print('Pickling %s.' % pickle_file_path)
        word_list, word_set = load_data(file_path, number=number_of_words, create_set=True)
        try:
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(word_list, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(word_set, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file_path, ':', e)
    return word_list, word_set


def calculate_test_accuracy(words, net, number, verbose=False, seed=None):
    if seed:
        random.seed(seed)
    positive = 0
    for i in range(0, number):
        word = random.choice(words)
        word_with_typo, _ = make_typo_char_replacement(word)
        predicted_word = net.correct_word(word_with_typo)
        if predicted_word == word:
            positive += 1
        else:
            if verbose:
                print(word_with_typo, '->', predicted_word, '(', word, ')')
    return (positive / number) * 100


class TypoTester:
    # def __init__(self, file_path, number_of_words=0):
    #     self.word_list, self.word_set = maybe_pickle(file_path, number_of_words=number_of_words)

    def __init__(self, word_list, num_tries_per_word=100, verbose=0, seed=None):
        self.word_list = word_list
        self.word_set = set(word_list)
        self.num_tries_per_word = num_tries_per_word
        self.verbosity_level = verbose
        if seed:
            random.seed(seed)

    def calculate_simple_accuracy(self, net):
        return calculate_test_accuracy(self.word_list, net, self.num_tries_per_word, self.verbosity_level > 0)

    def _is_correctly_predicted(self, predicted_word, original_word, place_of_typo):

        if predicted_word in self.word_set:
            if predicted_word[:place_of_typo]+predicted_word[place_of_typo+1:] == \
                            original_word[:place_of_typo]+original_word[place_of_typo+1:]:
                return True
        return False

    def calculate_fair_accuracy(self, net):
        total_tries = len(self.word_list) * self.num_tries_per_word

        ok_strict_counter = 0
        ok_fair_counter = 0

        for word in self.word_list:
            for i in range(self.num_tries_per_word):
                word_with_typo, place = make_typo_char_replacement(word)
                predicted_word = net.correct_word(word_with_typo)
                if predicted_word == word:
                    ok_strict_counter += 1
                    ok_fair_counter += 1
                elif self._is_correctly_predicted(predicted_word=predicted_word, original_word=word, place_of_typo=place):
                    ok_fair_counter += 1
                    if self.verbosity_level >= 2:
                        print(word_with_typo, '->', predicted_word, '(', word, ')')
                else:
                    if self.verbosity_level >= 3:
                        print(word_with_typo, '->', predicted_word, '(', word, ')')
        simple_accuracy, fair_accuracy = (ok_strict_counter / total_tries) * 100, (ok_fair_counter / total_tries) * 100
        if self.verbosity_level >= 1:
            print('Fair accuracy = {:.3f}%.'.format(fair_accuracy))
        return simple_accuracy, fair_accuracy


def vector_to_string(vector, string_length=None):
    char_vectors = np.hsplit(vector, max_word_length)
    result = ""
    for char_vector in char_vectors:
        result += int_to_char[np.argmax(char_vector)]
    if string_length is None:
        return result.rstrip()
    else:
        return result.rstrip().ljust(string_length)


def string_to_vector(string):
    string = string.ljust(max_word_length)
    result = np.zeros(word_vector_length, dtype=TC.floatX)
    for i, char in enumerate(string):
        result[i * nb_chars + char_to_int[char]] = 1.0
    return result


def vector_to_matrix2d(vector):
    return vector.reshape((nb_chars, -1), order='F')


def matrix2d_to_vector(matrix2d):
    return matrix2d.reshape(nb_chars*max_word_length, order='F')


def string_to_list_of_char_vectors(string):
    string = string.ljust(max_word_length)
    result = [np.zeros(nb_chars, dtype=TC.floatX) for _ in range(max_word_length)]
    for i, char in enumerate(string):
        result[i][char_to_int[char]] = 1.0
    return result


def make_typo_char_replacement(string):
    place = random.randrange(len(string))
    chars = [ch for ch in string]
    chars[place] = random.choice(int_to_char)
    return ''.join(chars), place


def data_generator(self, words, batch_size):
    """
Infinitely generates a batch of batch_size words, where X are words with typos
and Y are correct words. Pick batch_size as a integer divider of len(words).
    :param self:
    :param words:
    :param batch_size:
    """
    assert (len(words) / batch_size == round(len(words) / batch_size))
    nb_minibatches = int(len(words) / batch_size)
    while 1:
        random.shuffle(words)
        for i in range(nb_minibatches):
            x_data = np.empty((batch_size, self.input_size))
            y_data = np.empty((batch_size, self.input_size))
            for j in range(batch_size):
                word = words[i * batch_size + j]
                y_data[j, :] = string_to_vector(word)
                x_data[j, :] = string_to_vector(make_typo_char_replacement(word)[0])
            yield (x_data, y_data)


def data_generator_flexible(words, batch_size, add_correct_examples_to_input_data_flag=True,
                            examples_per_word=2, number_of_typos_per_typo_example=1, split_output_chars=False,
                            seed=None):
    """
Infinitely generates a batch of batch_size word examples, where X are words with typos (one may be correct)
and Y are correct words.
One epoch is after len(words)*(typos_per_word+add_correct_flag) examples.
Pick batch_size as a integer divider of this value.
    :param words: list of word strings
    :param batch_size: must be integer divider of len(words)*(examples_per_word)
    :param add_correct_examples_to_input_data_flag: if True correct input is also given in a dataset
    :param examples_per_word: how many examples (modified by typos + correct if specified) is given per one one word
    from dictionary
    :param number_of_typos_per_typo_example: how many typos in one noisy input example (noise level)
    :param split_output_chars: True, if multiple-softmax layer is used at the output of the net; otherwise False
    """
    if examples_per_word == 0:
        raise Exception('There must be at least one example per word given!')

    examples_in_one_epoch = len(words) * examples_per_word
    assert (examples_in_one_epoch / batch_size == round(examples_in_one_epoch / batch_size))

    nb_minibatches = int(examples_in_one_epoch / batch_size)

    if seed:
        random.seed(seed)

    while 1:
        random.shuffle(words)
        for batch_idx in range(nb_minibatches):
            x_data = np.empty((batch_size, word_vector_length))
            if split_output_chars:
                y_data = [np.empty((batch_size, nb_chars)) for _ in range(max_word_length)]
            else:
                y_data = np.empty((batch_size, word_vector_length))
            for example_idx_in_batch in range(batch_size):
                example_idx = batch_idx * batch_size + example_idx_in_batch
                example_idx_of_word = example_idx % examples_per_word
                word_idx = example_idx // examples_per_word
                word = words[word_idx]
                if add_correct_examples_to_input_data_flag and example_idx_of_word == 0:
                    x_data[example_idx_in_batch, :] = string_to_vector(word)
                else:
                    typo = word
                    for i in range(number_of_typos_per_typo_example):
                        typo, _ = make_typo_char_replacement(typo)
                    x_data[example_idx_in_batch, :] = string_to_vector(typo)
                if split_output_chars:
                    v = string_to_list_of_char_vectors(word)
                    for char_position, char_vector in enumerate(v):
                        y_data[char_position][example_idx_in_batch, :] = char_vector
                else:
                    y_data[example_idx_in_batch, :] = string_to_vector(word)
            yield (x_data, y_data)


def create_data_generator(word_file='data/words10000', batch_size=100, seed=1234):
    word_list, _ = maybe_pickle(word_file)
    generator = data_generator_flexible(word_list, batch_size, add_correct_examples_to_input_data_flag=True,
                                        examples_per_word=30, number_of_typos_per_typo_example=3,
                                        split_output_chars=True, seed=seed)
    return generator, word_list


class TypoAutoEncoder(DenoisingAutoEncoder):
    def __init__(self, word_vector_length, list_of_encoder_units_in_layers=[400]):
        super().__init__(word_vector_length, list_of_encoder_units_in_layers)
        self.name = "typo_autoencoder"

    def correct_word(self, word):
        x = np.empty((1, self.input_size))
        x[0, :] = string_to_vector(word)
        result = self.model.predict(x)
        return vector_to_string(result)


class DAEMultiSoftmax():
    def __init__(self, word_vector_length, list_of_encoder_units_in_layers=[300]):
        self.name = "dae_multi_softmax"
        self.input_size = word_vector_length
        input_layer = Input(shape=(word_vector_length,))

        encoder = input_layer
        for nb in list_of_encoder_units_in_layers:
            encoder = Dense(nb, activation='relu')(encoder)

        decoder = encoder
        if len(list_of_encoder_units_in_layers) > 1:
            for nb in reversed(list_of_encoder_units_in_layers[:-1]):
                decoder = Dense(nb, activation='relu')(decoder)

        output_layers = [Dense(nb_chars, activation='softmax')(decoder) for n in range(max_word_length)]

        self.model = Model(input=input_layer, output=output_layers)

    def compile(self):
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                           loss_weights=[1. for n in range(max_word_length)])

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch,
                                          callbacks=[]):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch, callbacks=callbacks)

    def correct_word(self, word):
        x = np.empty((1, self.input_size))
        x[0, :] = string_to_vector(word)
        result = self.model.predict(x)
        return vector_to_string(np.hstack(tuple(result)))


def test_net(model_file, weights_file, data_file, num_tries_per_word=100, verbose=0):
    tester = TypoTester(data_file, num_tries_per_word=num_tries_per_word, verbose=verbose)
    model, history = load_checkpoint(model_file, weights_file)
    dae = DAEMultiSoftmax(word_vector_length)
    dae.model = model
    dae.compile()
    accuracy, fair_accuracy = tester.calculate_fair_accuracy(dae)
    return accuracy, fair_accuracy


class TypoConvDAEMultiSoftMax(ConvDaeMultiSoftmax):
    def __init__(self, max_word_length_param, nb_chars_param, batch_norm=False):
        super().__init__(max_word_length_param, nb_chars_param, batch_norm=batch_norm)
        self.name = "typo_conv_multi_softmax_dae"

    def correct_word(self, word):
        x = np.empty((1, self.input_size))
        x[0, :] = string_to_vector(word)
        result = self.model.predict(x)
        return vector_to_string(np.hstack(tuple(result)))
