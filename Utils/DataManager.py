import numpy as np
import random


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
    result = np.zeros(word_vector_length, dtype=float)
    for i, char in enumerate(string):
        #print(i, char)
        result[i * nb_chars + char_to_int[char]] = 1.0
    return result

###### batch generation from data
def make_typo_char_replacement(string):
    place = random.randrange(len(string))
    chars = [ch for ch in string]
    chars[place] = random.choice(int_to_char)
    return ''.join(chars), place


def data_generator(words, batch_size, n_input):
    """
Infinitely generates a batch of batch_size words, where X are words with typos
and Y are correct words. Pick batch_size as a integer divider of len(words).
    :param self:
    :param words:
    :param batch_size:
    """
    print("pure division {}".format(len(words) / batch_size))
    print("rounded division {}".format(round(len(words) / batch_size)))

    assert (len(words) / batch_size == round(len(words) / batch_size))
    nb_minibatches = int(len(words) / batch_size)
    while 1:
        random.shuffle(words)
        for i in range(nb_minibatches):
            x_data = np.empty((batch_size, n_input))
            y_data = np.empty((batch_size, n_input))
            for j in range(batch_size):
                word = words[i * batch_size + j]
                y_data[j, :] = string_to_vector(word)
                x_data[j, :] = string_to_vector(make_typo_char_replacement(word)[0])
            #yield (x_data, y_data)
            yield y_data
