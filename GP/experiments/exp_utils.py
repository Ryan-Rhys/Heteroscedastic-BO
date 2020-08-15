# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script contains utility functions for running heteroscedastic Bayesian Optimisation experiments.
"""

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from neupy.layers import *
from neupy import algorithms
from neupy.init import HeNormal
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

from kernels import scipy_kernel
from mean_functions import zero_mean
from utils import posterior_predictive


def rmsle(expected, predicted):
    """
    Return the root mean squared log error between the predictions and the targets. Defined as in:
    http://neupy.com/2015/07/04/boston_house_prices_dataset.html#boston-house-price

    :param expected: target vector of size (N, 1) where N is the number of test examples
    :param predicted: prediction vector of size (N, 1) where N is the number of test examples
    :return: root mean square log error (penalises underpredictions relative to overpredictions)
    """

    log_expected = np.log1p(expected + 1)
    log_predicted = np.log1p(predicted + 1)
    squared_log_error = np.square(log_expected - log_predicted)
    return np.sqrt(np.mean(squared_log_error))


def train_network_lvq(num_neurons, mu, x_train, x_test, y_train, y_test, target_scaler):
    """
    Trains a neural network using the Levenberg-Marquardt optimiser. Importantly, re-initialises the network weights
    from scratch each time the function is called. Default weight initialisation in NeuPy is He initialisation. Single
    layer neural network with sigmoid activation functions.

    :param num_neurons: number of neurons in the network. Hyperparameter for BayesOpt
    :param mu: damping factor of LVQ optimisation algorithm. Hyperparameter for BayesOpt
    :param x_train: training inputs
    :param x_test: test inputs
    :param y_train: training targets
    :param y_test: test targets
    :param target_scalar: The scalar instance that has been used to scale the targets.
    :return: score: the root mean square log error on the test set.
    """

    # Create a single layer nueral network with Sigmoid activations. We seed the HeNormal weight initialiser for reproducibility

    network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)), Sigmoid(y_test.shape[1], weight=HeNormal(seed=1)))

    optimizer = algorithms.LevenbergMarquardt(network, mu=mu, verbose=False, show_epoch=5)
    optimizer.train(x_train, y_train, x_test, y_test, epochs=30)

    y_predict = optimizer.predict(x_test).round(1)

    # We assume the targets have been scaled by the MinMaxScaler and hence we take the inverse transform to measure the loss.

    score = rmsle(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(y_predict))

    return score


def train_network_conj_grad(num_neurons, num_layers, x_train, x_test, y_train, y_test):
    """
    Trains a neural network using the Conjugate Gradients optimiser. Importantly, re-initialises the network weights
    from scratch each time the function is called. Default weight initialisation in NeuPy is He initialisation. We vary
    the number of neurons and the number of layers.

    :param num_neurons: number of neurons in the network.
    :param num_layers: the number of layers in the network. Max value of 3 layers.
    :param x_train: training inputs
    :param x_test: test inputs
    :param y_train: training targets
    :param y_test: test targets
    :return: score: the root mean square log error on the test set.
    """

    assert (num_layers <= 3), "The number of layers can't exceeed 3."

    if num_layers == 1:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                       Sigmoid(y_test.shape[1], weight=HeNormal(seed=1)))
    elif num_layers == 2:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                       Sigmoid(num_neurons, weight=HeNormal(seed=1)), Sigmoid(y_test.shape[1],
                                                                              weight=HeNormal(seed=1)))
    else:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                       Sigmoid(num_neurons, weight=HeNormal(seed=1)), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                       Sigmoid(y_test.shape[1], weight=HeNormal(seed=1)))

    optimizer = algorithms.ConjugateGradient(network, loss='binary_crossentropy', verbose=False, show_epoch=5)

    import time
    start = time.time()
    optimizer.train(x_train, y_train, x_test, y_test, epochs=10)
    end = time.time()
    print('training time is: ' + str(end - start))
    y_predict = optimizer.predict(x_test).round(0)
    score = metrics.accuracy_score(y_test, y_predict)

    return score


def train_network_adam(num_neurons, num_layers, x_train, x_test, y_train, y_test):
    """
    Trains a neural network using the Adam optimiser. Importantly, re-initialises the network weights
    from scratch each time the function is called. Default weight initialisation in NeuPy is He initialisation. We vary
    the number of neurons and the number of layers.

    :param num_neurons: number of neurons in the network.
    :param num_layers: the number of layers in the network. Max value of 3 layers.
    :param x_train: training inputs
    :param x_test: test inputs
    :param y_train: training targets
    :param y_test: test targets
    :return: score: the root mean square log error on the test set.
    """

    assert (num_layers <= 3), "The number of layers can't exceeed 3."

    if num_layers == 1:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                       Sigmoid(y_test.shape[1], weight=HeNormal(seed=1)))
    elif num_layers == 2:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                        Sigmoid(num_neurons, weight=HeNormal(seed=1)), Sigmoid(y_test.shape[1],
                                                                               weight=HeNormal(seed=1)))
    else:
        network = join(Input(x_train.shape[1]), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                        Sigmoid(num_neurons, weight=HeNormal(seed=1)), Sigmoid(num_neurons, weight=HeNormal(seed=1)),
                        Sigmoid(y_test.shape[1], weight=HeNormal(seed=1)))

    optimizer = algorithms.Adam(network, loss='binary_crossentropy', verbose=False)

    optimizer.train(x_train, y_train, x_test, y_test, epochs=10)
    y_predict = optimizer.predict(x_test).round(0)
    score = metrics.accuracy_score(y_test, y_predict)

    return score


def train_network_keras_adam(num_neurons, num_layers, x_train, x_test, y_train, y_test):
    """
    Trains a neural network using the Adam optimiser and the keras api. Importantly, re-initialises the network weights
    from scratch each time the function is called. Default weight initialisation in NeuPy is He initialisation. We vary
    the number of neurons and the number of layers.

    :param num_neurons: number of neurons in the network.
    :param num_layers: the number of layers in the network. Max value of 1 layer.
    :param x_train: training inputs
    :param x_test: test inputs
    :param y_train: training targets
    :param y_test: test targets
    :return: score: the root mean square log error on the test set.
    """

    assert (num_layers <= 3), "The number of layers can't exceeed 3."

    model = Sequential()
    model.add(Dense(num_neurons, input_dim=x_train.shape[1], activation='sigmoid', kernel_initializer='he_uniform'))

    if num_layers >= 2:
        model.add(Dense(num_neurons, input_dim=x_train.shape[1], activation='sigmoid', kernel_initializer='he_uniform'))
    if num_layers >= 3:
        model.add(Dense(num_neurons, input_dim=x_train.shape[1], activation='sigmoid', kernel_initializer='he_uniform'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, verbose=0)

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return accuracy


def measure_reg_performance(data, target, num_neurons, mu, k_folds, n_trials, target_scaler):
    """
    Assess regression performance and performance noise across n independent trials for a fixed set of network
    parameters. The noise is taken to be the performance standard deviation across the k validation folds of k-fold
    cross-validation.

    :param data: The inputs (N, d)
    :param target: The target (N, 1)
    :param num_neurons: The number of neurons in the hidden layer.
    :param mu: The damping factor in Levenberg-Marquardt Optimisation
    :param k_folds: The number of folds of cross-validation on which to gauge the noise
    :param n_trials: The number of independent initialisations of k-fold cross-validation (randomises the validation sets)
    :param target_scaler: The scaler that has been applied to the target values.
    :return: mean performance and mean noise.
    """

    trial_performances = [] # List of performances (e.g. rmsle values) for the n trials
    trial_noises = [] # List of noises (e.g. std of performance) for the n trials

    for random_number in range(1, n_trials + 1):

        errors = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_number)

        for train, test in kf.split(data):

            error = train_network_lvq(num_neurons, mu, data[train], data[test], target[train], target[test], target_scaler)
            errors.append(error)

        performance_mean = np.mean(errors)
        noise = np.std(errors)  # noise

        print('mean rmsle across K folds:' + str(performance_mean))
        print('standard_deviation:' + str(noise))

        trial_performances.append(performance_mean)
        trial_noises.append(noise)

    av_trial_performance = np.mean(trial_performances)
    av_trial_noise = np.mean(trial_noises)

    return av_trial_performance, av_trial_noise


def measure_class_performance(data, target, num_neurons, num_layers, k_folds, n_trials):
    """
    Assess classification performance and performance noise across n independent trials for a fixed set of network
    parameters. The noise is taken to be the performance standard deviation across the k validation folds of k-fold
    cross-validation. There is no target scaler needed for classification.

    :param data: The inputs (N, d)
    :param target: The target (N, 1)
    :param num_neurons: The number of neurons in the hidden layer.
    :param num_layers: The number of layers in the network.
    :param k_folds: The number of folds of cross-validation on which to gauge the noise
    :param n_trials: The number of independent initialisations of k-fold cross-validation (randomises the validation sets)
    :return: mean performance and mean noise.
    """

    trial_performances = [] # List of performances (e.g. accuracy values) for the n trials
    trial_noises = [] # List of noises (e.g. std of performance) for the n trials

    for random_number in range(1, n_trials + 1):

        errors = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_number)

        for train, test in kf.split(data):
            error = train_network_keras_adam(num_neurons, num_layers, data[train], data[test], target[train], target[test])
            errors.append(error)

        performance_mean = np.mean(errors)*100 # multiplying by 100 converts fraction into percentage.
        noise = np.std(errors)*100  # noise

        print('mean accuracy across {} folds:'.format(k_folds) + str(performance_mean))
        #print('standard_deviation:' + str(noise))

        trial_performances.append(performance_mean)
        trial_noises.append(noise)

    av_trial_performance = np.mean(trial_performances)
    av_trial_noise = np.mean(trial_noises)

    return av_trial_performance, av_trial_noise
