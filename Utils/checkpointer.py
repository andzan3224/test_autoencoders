import numpy as np
import pickle
from keras.callbacks import Callback
from keras.models import model_from_json


class ModelCheckpointingNoHDF5(Callback):
    def __init__(self, net, epoch_modulo=10, prev_history=None, file_path='models/', tester=None):
        """

        :param epoch_modulo: After what number of epochs checkpointing should take place
        :param prev_history: numpy array of the form : [[epoch, loss, test_accuracy], [epoch, loss, test_accuracy]]
        :param tester: typodomain::TypoTester object
        """
        super().__init__()
        self.net = net
        self.epoch_modulo = epoch_modulo
        if prev_history is not None:
            self.history = [tuple(row) for row in prev_history]
            self.best_test_accuracy = np.max(prev_history[:,2])
            self.epoch = prev_history[-1][0]
        else:
            self.history = []
            self.best_test_accuracy = 0.
            self.epoch = 0
        self.file_path = file_path
        self.tester = tester
        self.do_tests = True if tester is not None else False

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        if self.epoch % self.epoch_modulo == 0:
            print('\nSaving weights of the model...')
            list_with_weights = self.model.get_weights()
            output = open(self.file_path+'last_model.pkl', 'wb')
            pickle.dump(list_with_weights, output, pickle.HIGHEST_PROTOCOL)
            if self.do_tests:
                print('Testing the model...')
                _, test_accuracy = self.tester.calculate_fair_accuracy(net=self.net)            # this takes some time
                print('Test accuracy:', test_accuracy, ' - best.' if test_accuracy > self.best_test_accuracy else '.')
            else:
                test_accuracy = None
            self.history.append((self.epoch, logs.get('loss'), test_accuracy))
            pickle.dump(np.asarray(self.history), output, pickle.HIGHEST_PROTOCOL)
            output.close()

            if self.do_tests and test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = test_accuracy
                print('Saving the model.')
                output = open(self.file_path+'best_model.pkl', 'wb')
                pickle.dump(list_with_weights, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(np.asarray(self.history), output, pickle.HIGHEST_PROTOCOL)
                output.close()


def load_checkpoint(filepath_to_arch, filepath_to_weights):
    model = model_from_json(open(filepath_to_arch, 'r').read())
    pkl_file = open(filepath_to_weights, 'rb')
    weights = pickle.load(pkl_file)
    history = np.asarray(pickle.load(pkl_file))
    pkl_file.close()
    model.set_weights(weights)
    return model, history


# def test():
#     words = load_data('data/words1000')
#     dae = DAEMultiSoftmax(word_vector_length, [300])
#     dae.compile()
#
#     # save architecture
#     model_file_name = 'models/test_model_arch.json'
#     json_string = dae.model.to_json()
#     open(model_file_name, 'w').write(json_string)
#
#     train_data_generator = data_generator_flexible(words=words, batch_size=100,
#                                                    add_correct_examples_to_input_data_flag=True,
#                                                    examples_per_word=30,
#                                                    number_of_typos_per_typo_example=3,
#                                                    split_output_chars=True)
#
#     checkpointer = ModelCheckpointingNoHDF5(dae, 5, file_path='models/test_', test_words=words)
#
#     dae.train_using_data_generator_simple(train_data_generator, nb_train_samples_per_epoch=100*len(words),
#                                           nb_epoch=30, callbacks=[checkpointer])
#
#
# def test_load(do_posttesting=False):
#     words = load_data('data/words10000')
#     dae = DAEMultiSoftmax(word_vector_length, [300])
#     model_file_name = 'models/apl10_exp5_model_arch.json'
#     best_weights_path = 'models/apl10_exp5_best_model.pkl'
#
#     model, history = load_checkpoint(model_file_name, best_weights_path)
#     dae.model = model
#     dae.compile()
#
#     print(history)
#     if do_posttesting:
#         print('test accuracy: ', dae.test_me(words, 10000))
#
#     return dae, history
#
#
# def test_continuation():
#     words = load_data('data/words1000')
#     dae, history = test_load()
#
#     train_data_generator = data_generator_flexible(words=words, batch_size=100,
#                                                    add_correct_examples_to_input_data_flag=True,
#                                                    examples_per_word=30,
#                                                    number_of_typos_per_typo_example=3,
#                                                    split_output_chars=True)
#
#     checkpointer = ModelCheckpointingNoHDF5(dae, 4, file_path='models/test_', test_words=words, prev_history=history)
#
#     dae.train_using_data_generator_simple(train_data_generator, nb_train_samples_per_epoch=100*len(words),
#                                           nb_epoch=20, callbacks=[checkpointer])
