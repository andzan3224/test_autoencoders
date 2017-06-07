checkpointer.py# coding=utf-8
import math

from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json


class NeuralNetwork:
    def __init__(self, input_size, output_size=1):
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = 10000
        self.batch_size = 100
        self.model = Sequential()
        self.name = "undefined"

    def save_net(self):
        json_string = self.model.to_json()
        open('models/keras_{}.json'.format(self.name), 'w').write(json_string)
        self.model.save_weights('keras_{}.h5'.format(self.name), overwrite=True)

    def load_net(self):
        self.model = model_from_json(open('models/keras_{}.json'.format(self.name)).read())
        self.model.load_weights('models/keras_{}.h5'.format(self.name))

    def train_full_dataset(self, x_data, y_data, x_valid, y_valid, nb_epoch = 1000):
        early_stopping = EarlyStopping(patience=42)
        self.model.fit(x_data, y_data, nb_epoch=nb_epoch, batch_size=self.batch_size,
                       validation_data=(x_valid, y_valid), callbacks=[early_stopping])

    def train_using_data_generator(self, train_data_generator, valid_data_generator,
                                   nb_epoch, nb_train_samples_per_epoch, nb_valid_samples_per_epoch, patience=42):
        cbs = []
        if patience>0:
            cbs.append(EarlyStopping(patience=patience))
        history = self.model.fit_generator(train_data_generator,
                                           samples_per_epoch=nb_train_samples_per_epoch, nb_epoch=nb_epoch,
                                           validation_data=valid_data_generator, nb_val_samples=nb_valid_samples_per_epoch,
                                           callbacks=cbs)
        return history

    def train_using_data_generator_simple(self, train_data_generator, nb_train_samples_per_epoch, nb_epoch):
        return self.model.fit_generator(train_data_generator, samples_per_epoch=nb_train_samples_per_epoch,
                                        nb_epoch=nb_epoch)
