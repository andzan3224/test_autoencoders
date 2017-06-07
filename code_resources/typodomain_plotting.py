import matplotlib.pyplot as plt
import numpy as np

from Utils.typodomain import nb_chars, max_word_length, vector_to_string, matrix2d_to_vector, int_to_char


# import matplotlib
# matplotlib.rc('font', family='')


def show_word_image(image_data):

    if image_data.shape == (nb_chars, max_word_length):
        yaxis = 'alphabet'
    elif image_data.shape == (max_word_length, nb_chars):
        yaxis = 'word'
    else:
        raise Exception('Bad image_data shape')

    plt.matshow(image_data)

    if yaxis == 'alphabet':
        word_as_label = vector_to_string(matrix2d_to_vector(image_data), max_word_length)
        plt.yticks(np.arange(nb_chars), int_to_char)
        plt.xticks(np.arange(max_word_length), word_as_label)
    else:
        word_as_label = vector_to_string(matrix2d_to_vector(image_data.transpose()), max_word_length)
        plt.xticks(np.arange(nb_chars), int_to_char)
        plt.yticks(np.arange(max_word_length), word_as_label)

    plt.show()
