from keras.engine import Model
from keras.layers import Input, Dense, Reshape, Convolution1D, MaxPooling1D, Flatten, Activation, BatchNormalization

from Utils.neuralnetwork import NeuralNetwork


class DenoisingAutoEncoder(NeuralNetwork):
    def __init__(self, input_size, list_of_encoder_units_in_layers=[400]):
        super().__init__(input_size)
        self.name = "dae"
        input_layer = Input(shape=(self.input_size,))

        encoder = input_layer
        for nb in list_of_encoder_units_in_layers:
            encoder = Dense(nb, activation='relu')(encoder)

        decoder = encoder
        if len(list_of_encoder_units_in_layers) > 1:
            for nb in reversed(list_of_encoder_units_in_layers[-1]):
                decoder = Dense(nb, activation='relu')(decoder)

        decoder = Dense(self.input_size, activation='sigmoid')(decoder)

        self.model = Model(input=input_layer, output=decoder)

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adadelta')

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch,
                                          callbacks=[]):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch, callbacks=callbacks)


class ConvDaeMultiSoftmax:
    """
    Convolutional denoising autoencoder that transforms [batch_size, input_height*input_width] data to
    [batch_size, input_height, input_width] and used convolutional and maxpooling layers and outputs
    a multi-softmax layer: softmax(size=input_width) concatenated input_height times
    """
    def __init__(self, input_height, input_width, batch_norm=False):
        self.name = "cnn_multi_softmax"
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_height * input_width

        input_layer = Input(shape=(self.input_size, ))
        reshaped_input = Reshape((input_height, input_width))(input_layer)      # output (batch_size, input_height, input_width) (bs, max_word_length = 16, nb_chars)
        encoder = Convolution1D(64, 3, border_mode='same', init='he_normal')(reshaped_input)  # output (batch_size, input_height, 64)
        if batch_norm:
            encoder = BatchNormalization(mode=0, axis=2)(encoder)
        encoder = Activation('relu')(encoder)
        encoder = MaxPooling1D(pool_length=4, stride=2, border_mode='valid')(encoder)     # output (batch_size, 7, 64)

        encoder = Convolution1D(150, 3, border_mode='same', init='he_normal')(encoder)      # output (batch_size, 7, 150)
        if batch_norm:
            encoder = BatchNormalization(mode=0, axis=2)(encoder)
        encoder = Activation('relu')(encoder)
        encoded = MaxPooling1D(pool_length=3, stride=2, border_mode='valid')(encoder)     # output (batch_size, 3, 150)

        flattened_code = Flatten()(encoded)                 # output (batch_size, 450)
        # at this point the representation is 450-dimensional

        output_layers = [Dense(input_width, activation='softmax')(flattened_code) for _ in range(input_height)]

        self.model = Model(input=input_layer, output=output_layers)

        # for testing:
        self.reshaped_input = Model(input=input_layer, output=reshaped_input)

        # separate encoder part:
        self.encoder = Model(input=input_layer, output=encoded)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           loss_weights=[1. for _ in range(self.input_height)])

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch,
                                          callbacks=[]):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch, callbacks=callbacks)


class BatchNormalizationTester:
    """
    Convolutional denoising autoencoder that transforms [batch_size, input_height*input_width] data to
    [batch_size, input_height, input_width] and used convolutional and maxpooling layers and outputs
    a multi-softmax layer: softmax(size=input_width) concatenated input_height times
    """
    def __init__(self, input_height, input_width):
        self.name = "batch_normalization_tester"
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_height * input_width

        # TODO learn this layer by layer
        input_layer = Input(shape=(self.input_size, ))
        reshaped_input = Reshape((input_height, input_width))(input_layer)      # output (batch_size, input_height, input_width) (bs, max_word_length = 16, nb_chars)
        encoder = Convolution1D(64, 3, border_mode='same', init='he_normal')(reshaped_input)  # output (batch_size, input_height, 64)
        encoder_bn = BatchNormalization(mode=2, axis=2)(encoder)
        # encoder = Activation('relu')(encoder)
        # encoder = MaxPooling1D(pool_length=4, stride=2, border_mode='valid')(encoder)     # output (batch_size, 7, 64)

        self.model = Model(input=input_layer, output=encoder_bn)

        # for testing:
        self.before_bn = Model(input=input_layer, output=encoder)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch,
                                          callbacks=[]):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch, callbacks=callbacks)


def _block(input_, nb_filters, filter_width=3, pool_length=2, batch_norm=True):
    output = Convolution1D(nb_filters, filter_width, border_mode='same', init='he_normal')(input_)
    if batch_norm:
        output = BatchNormalization(mode=0, axis=2)(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(pool_length=pool_length, stride=pool_length, border_mode='valid')(output)
    return output


class ConvDaeMultiSoftmax2:
    """
    Convolutional denoising autoencoder that transforms [batch_size, input_height*input_width] data to
    [batch_size, input_height, input_width] and used convolutional and maxpooling layers and outputs
    a multi-softmax layer: softmax(size=input_width) concatenated input_height times
    """
    def __init__(self, input_height, input_width, nb_filters_list=[128, 256, 512], batch_norm=False):
        self.name = "cnn_2_multi_softmax"
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_height * input_width
        input_layer = Input(shape=(self.input_size, ))
        reshaped_input = Reshape((input_height, input_width))(input_layer)      # output (batch_size, input_height, input_width) (bs, max_word_length = 16, nb_chars)

        encoded = reshaped_input
        for nb_filters in nb_filters_list:
            encoded = _block(encoded, nb_filters, batch_norm=batch_norm)        # output (batch_size, input_height, nb_filters)

        flattened_code = Flatten()(encoded)                 # output (batch_size, input_height * nb_filters_list[-1])

        output_layers = [Dense(input_width, activation='softmax')(flattened_code) for _ in range(input_height)]

        self.model = Model(input=input_layer, output=output_layers)

        # for testing:
        self.reshaped_input = Model(input=input_layer, output=reshaped_input)

        # separate encoder part:
        self.encoder = Model(input=input_layer, output=encoded)



    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           loss_weights=[1. for _ in range(self.input_height)])

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch,
                                          callbacks=[]):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch, callbacks=callbacks)
