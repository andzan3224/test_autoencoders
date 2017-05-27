import sys

import numpy as np

from Utils.typodomain import TypoAutoEncoder, word_vector_length, data_generator_flexible,\
    calculate_test_accuracy, load_data

# from keras.utils.visualize_util import plot


variants = [(True, 1), (True, 2), (True, 3), (True, 5), (True, 10), (True, 30), (True, 100), (False, 1)]
# percent_of_examples_with_typos
variants_labels = ['0%', '50%', '66%', '80%', '90%', '97%', '99%', '100%']

# number of typos in ONE word example
variants_part2 = [(True, 2), (True, 3), (True, 5), (True, 10), (False, 2), (False, 3), (False, 5), (False, 10)]
variants_labels_part2 = ['2', '3', '5', '10']

variants_part3 = [(30, 1), (30, 2), (30, 3), (30, 5), (100, 1), (100, 2), (100, 3), (100, 5)]


def single_run(words, variant, nb_epochs=100, batch_size=100):

    correct_example_in_input_data_flag = variant[0]
    examples_per_word = variant[1]

    dae = TypoAutoEncoder(word_vector_length, [300])
    dae.compile()

    train_data_generator = data_generator_flexible(words=words, batch_size=batch_size,
                                                   add_correct_examples_to_input_data_flag=
                                                   correct_example_in_input_data_flag,
                                                   examples_per_word=examples_per_word)

    history = dae.train_using_data_generator_simple(train_data_generator,
                                                    nb_train_samples_per_epoch=100*len(words),
                                                    nb_epoch=nb_epochs)

    sign = '+' if correct_example_in_input_data_flag is True else '-'
    np.save('results/exp1b_accuracy_' + str(examples_per_word) + sign +'_' + str(len(words)),
            calculate_test_accuracy(words, dae, 10000, False))
    np.save('results/exp1b_loss_history_'+str(examples_per_word)+sign+'_'+str(len(words)),
            np.asarray(history.history['loss']))


def single_run_part2(words, variant, nb_epochs=100, batch_size=100):

    correct_example_in_input_data_flag = variant[0]
    typos_in_one_word_example = variant[1]

    dae = TypoAutoEncoder(word_vector_length, [300])
    dae.compile()

    train_data_generator = data_generator_many_typos_in_one_example(
        words=words, batch_size=batch_size, number_of_typos_in_one_word=typos_in_one_word_example,
        add_correct_examples_to_input_data_flag=correct_example_in_input_data_flag)

    history = dae.train_using_data_generator_simple(train_data_generator,
                                                    nb_train_samples_per_epoch=100*len(words),
                                                    nb_epoch=nb_epochs)

    sign = '+' if correct_example_in_input_data_flag is True else '-'
    np.save('results/exp1_part2_accuracy_' + str(typos_in_one_word_example) + sign +'_' + str(len(words)), calculate_test_accuracy(words, dae, 10000, False))
    np.save('results/exp1_part2_loss_history_'+str(typos_in_one_word_example)+sign+'_'+str(len(words)), np.asarray(history.history['loss']))


def single_run_part3(words, variant):

    examples_per_word = variant[0]
    typos_in_one_word_example = variant[1]

    dae = TypoAutoEncoder(word_vector_length, [300])
    dae.compile()

    train_data_generator = data_generator_flexible(words=words, batch_size=100,
                                                   add_correct_examples_to_input_data_flag=True,
                                                   examples_per_word=examples_per_word,
                                                   number_of_typos_per_typo_example=typos_in_one_word_example)

    accuracy_history = np.zeros(100)
    for i in range(len(accuracy_history)):
        print('Epochs ' + str(i*10+1) + '-' + str((i+1)*10))
        dae.train_using_data_generator_simple(train_data_generator, nb_train_samples_per_epoch=100*len(words),
                                              nb_epoch=10)
        accuracy_history[i] = calculate_test_accuracy(words, dae, 10000, False)

    np.save('results/exp1_part3_accuracy_'+str(examples_per_word)+'_'+str(typos_in_one_word_example)+'_'+str(len(words)), accuracy_history)


def do_experiment(words, variants_idxs, nb_epochs=1000):
    print('Dataset with', len(words), 'words will be used.')
    for idx in variants_idxs:
        single_run(words, variants[int(idx)], nb_epochs)


def do_experiment_part2(words, variants_idxs, nb_epochs=1000):
    print('Dataset with', len(words), 'words will be used.')
    for idx in variants_idxs:
        single_run_part2(words, variants_part2[int(idx)], nb_epochs)


def do_experiment_part3(words, variants_idxs):
    print('Dataset with', len(words), 'words will be used.')
    for idx in variants_idxs:
        single_run_part3(words, variants_part3[int(idx)])


if __name__ == "__main__":
    words = load_data('data/words1000')
    #do_experiment(words, sys.argv[1:], 1000)
    # do_experiment_part2(words, sys.argv[1:], 1000)
    do_experiment_part3(words, sys.argv[1:])


