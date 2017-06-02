from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########### spelling corrector

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
        print(i, char)
        result[i * nb_chars + char_to_int[char]] = 1.0
    return result

words = load_data('data/words100')
print(words)

for j in range(3): #len(words[0])):
    print((words[0])[j])
    a = string_to_vector( (words[0])[j])
    print (a)
    for i, ch in enumerate(a):
        print(i,ch)



import sys
sys.exit(0)


########################### MNIST ###################

#import load_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 64 # 1st layer num features
n_hidden_2 = 32 # 2nd layer num features
n_input_t = 784 # MNIST data input (img shape: 28*28)
n_input = n_input_t // 2

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# tf Graph input (only pictures)
Xt = tf.placeholder("float", [batch_size, n_input_t])
print("Shape of Xt is {}".format(tf.shape(Xt)))

X = tf.slice(Xt,[101,0],[1,n_input])
print("Shape of X is {}".format(tf.shape(X)))


# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batch_xs, batch_ys = mnist.train.next_batch(batch_size)
t , c  = sess.run([Xt, X], feed_dict={Xt: batch_xs})
print(c)

plt.imshow(np.reshape(c,(14,28)))
plt.draw()
plt.waitforbuttonpress()


