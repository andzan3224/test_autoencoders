from __future__ import division, print_function, absolute_import
import tensorflow as tf
import Utils.DataManager as ud


words = ud.load_data('data/words1000')
# print(words)


###################### model
# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 20
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 64  # 1st layer num features
n_hidden_2 = 32  # 2nd layer num features
n_input = 576  # Single word data input (img shape: 16 ch per single word X 36 symbols in 1 hot encoding)

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

# generator for the batch to feed into the computation
batch = ud.data_generator(words[0], batch_size, n_input)


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


import sys
sys.exit(0)

############## not appl
plt.imshow(np.reshape(c,(14,28)))
plt.draw()
plt.waitforbuttonpress()


