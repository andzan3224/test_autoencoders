from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

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
        #print(i, char)
        result[i * nb_chars + char_to_int[char]] = 1.0
    return result

words = load_data('data/words1000')
# print(words)


# for j in range(1): #len(words[0])):
#     print((words[0])[j])
#     a = string_to_vector( (words[0])[j] )
#     print ("string to vector conversion is {}".format(a))
#     for i, ch in enumerate(a):
#         print(i,ch)


###################### model
# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 20
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 64 # 1st layer num features
n_hidden_2 = 32 # 2nd layer num features
n_input = 576 # Single word data input (img shape: 16 ch per single word X 36 symbols in 1 hot encoding)

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


###### batch generation from data
def make_typo_char_replacement(string):
    place = random.randrange(len(string))
    chars = [ch for ch in string]
    chars[place] = random.choice(int_to_char)
    return ''.join(chars), place


def data_generator(words, batch_size):
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


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# tf Graph input (only words)
X = tf.placeholder("float", [batch_size, n_input])
#print("Shape of Xt is {}".format(tf.shape(Xt)))

#X = tf.slice(Xt,[9,0],[1,n_input])
#print("Shape of X is {}".format(tf.shape(X)))


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


batch = data_generator(words[0], batch_size)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = 15
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch1 = next(batch)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch1})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")


#t , c  = sess.run([Xt, X], feed_dict={Xt: batch1}) #this should be a list obtained from the generator
#print(c)



import sys
sys.exit(0)

############## not appl
plt.imshow(np.reshape(c,(14,28)))
plt.draw()
plt.waitforbuttonpress()


